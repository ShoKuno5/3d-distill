#!/usr/bin/env python3
"""Distillation training entry point.

Usage:
    cd /mnt/workspace/kuno/distillation

    # Single GPU
    envs/hunyuan3d-venv/bin/python \
        distill_methods/src/train.py \
        --config distill_methods/config.yaml \
        --method pd

    # Multi-GPU (4x L20X)
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
        distill_methods/src/train.py \
        --config distill_methods/config.yaml \
        --method pd --stage 0
"""

import argparse
import logging
import os
import sys
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import yaml

logger = logging.getLogger(__name__)

# Ensure experiment src is importable
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

class LatentDataset(Dataset):
    """Pre-encoded VAE latents + image conditions (from prepare_training_data.py)."""

    def __init__(self, data_dir: str):
        self.files = sorted(glob(os.path.join(data_dir, "*.npz")))
        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {data_dir}")
        logger.info("LatentDataset: %d samples from %s", len(self.files), data_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True)
        return {
            "latent": torch.from_numpy(data["latent"].astype(np.float32)),
            "image_cond": torch.from_numpy(data["image_cond"].astype(np.float32)),
            "object_id": str(data["object_id"]),
        }


def make_dataloader(dataset: Dataset, config: dict, batch_size_override: int | None = None) -> DataLoader:
    """Create DataLoader with optional DistributedSampler."""
    train_cfg = config["training"]
    batch_size = batch_size_override or train_cfg["batch_size"]
    sampler = None
    shuffle = True
    if dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=True)
        shuffle = False

    # drop_last only when dataset is large enough
    drop_last = len(dataset) >= batch_size * 2

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
        drop_last=drop_last,
    )


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Distillation training")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument(
        "--method", required=True,
        choices=["pd", "cd", "dmd1", "dmd2"],
        help="Distillation method",
    )
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--stage", type=int, default=0, help="PD stage index (0-based)")
    args = parser.parse_args()

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # DDP init
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        logger.info("DDP rank=%d local_rank=%d", dist.get_rank(), local_rank)

    # Config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Seed
    seed = config["training"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Dataset
    dataset = LatentDataset(config["training"]["training_data_dir"])
    dataloader = make_dataloader(dataset, config)

    # Method dispatch
    from progressive_distillation import ProgressiveDistillation
    from consistency_distillation import ConsistencyDistillation
    from dmd1_distillation import DMD1Distillation
    from dmd2_distillation import DMD2Distillation

    method_cls = {
        "pd": ProgressiveDistillation,
        "cd": ConsistencyDistillation,
        "dmd1": DMD1Distillation,
        "dmd2": DMD2Distillation,
    }[args.method]

    distiller = method_cls(config)

    if args.resume:
        distiller.load_checkpoint(args.resume)

    # Method-specific setup
    method_cfg = config["training"]["methods"][args.method]
    # total_steps: method-level override > global default
    global_total_steps = config["training"]["total_steps"]

    if args.method == "pd":
        # For stage > 0, load the merged model from the previous stage
        if args.stage > 0:
            prev_merged = os.path.join(
                config["output_root"], "checkpoints", "pd",
                f"stage_{args.stage - 1}_merged", "model.pt",
            )
            if not os.path.exists(prev_merged):
                raise FileNotFoundError(
                    f"PD stage {args.stage} requires merged model from stage "
                    f"{args.stage - 1}, but {prev_merged} not found. "
                    f"Run stage {args.stage - 1} first."
                )
            distiller.load_merged_base(prev_merged)

        distiller.set_stage(args.stage)
        total_steps = method_cfg["steps_per_stage"]
    else:
        total_steps = method_cfg.get("total_steps", global_total_steps)

    distiller.train(dataloader, total_steps=total_steps)

    # For PD, save merged model after each stage for next stage handoff
    if args.method == "pd" and distiller.is_main:
        distiller.save_merged_model(args.stage)

    # Cleanup DDP
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
