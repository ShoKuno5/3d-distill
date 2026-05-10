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
from datetime import datetime
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import yaml

# Wandb API key: set via WANDB_API_KEY env var or `wandb login`

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


def assert_no_toys4k_in_train(dataset: Dataset) -> None:
    """Hard-fail if any training object_id matches a Toys4k asset.

    Toys4k is reserved for evaluation against the teacher and must never
    enter the distillation training distribution. Object IDs are derived
    from the npz filename (matches what prepare_training_data.py writes).
    Two reference lists are checked:
      data/toys4k_uids.txt   — TRELLIS-500K SHA-256 of every Toys4k asset
      data/toys4k_names.txt  — legacy 'category_NNN' names from the
                               Toys4k metadata file_identifier column
    Either file missing or any overlap is a hard failure.
    """
    data_dir = _SRC_DIR.parent / "data"
    uids_file = data_dir / "toys4k_uids.txt"
    names_file = data_dir / "toys4k_names.txt"
    if not uids_file.exists() or not names_file.exists():
        raise FileNotFoundError(
            f"Toys4k guard files missing: expected both {uids_file} and "
            f"{names_file}. Cannot proceed — Toys4k must stay eval-only."
        )

    toys4k_uids = {ln.strip() for ln in uids_file.read_text().splitlines() if ln.strip()}
    toys4k_names = {ln.strip() for ln in names_file.read_text().splitlines() if ln.strip()}
    train_ids = {Path(f).stem for f in dataset.files}

    uid_overlap = train_ids & toys4k_uids
    name_overlap = train_ids & toys4k_names
    if uid_overlap or name_overlap:
        raise AssertionError(
            f"Toys4k contamination in training set: "
            f"{len(uid_overlap)} sha256 collisions, "
            f"{len(name_overlap)} name collisions. Toys4k is eval-only.\n"
            f"  sha256 examples: {sorted(uid_overlap)[:5]}\n"
            f"  name examples:   {sorted(name_overlap)[:5]}"
        )

    logger.info(
        "Toys4k guard OK: %d train ids checked vs %d sha256 + %d name refs, no overlap",
        len(train_ids), len(toys4k_uids), len(toys4k_names),
    )


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
        choices=["pd", "cd", "dmd1", "dmd2", "sid"],
        help="Distillation method",
    )
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--stage", type=int, default=0, help="PD stage index (0-based)")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output_root from config")
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

    if args.output_dir:
        config["output_root"] = args.output_dir

    # Seed
    seed = config["training"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Dataset
    dataset = LatentDataset(config["training"]["training_data_dir"])
    assert_no_toys4k_in_train(dataset)
    dataloader = make_dataloader(dataset, config)

    # Method dispatch
    from progressive_distillation import ProgressiveDistillation
    from consistency_distillation import ConsistencyDistillation
    from dmd1_distillation import DMD1Distillation
    from dmd2_distillation import DMD2Distillation
    from sid_distillation import SiDDistillation

    method_cls = {
        "pd": ProgressiveDistillation,
        "cd": ConsistencyDistillation,
        "dmd1": DMD1Distillation,
        "dmd2": DMD2Distillation,
        "sid": SiDDistillation,
    }[args.method]

    distiller = method_cls(config)

    resume_step = 0
    if args.resume:
        resume_step = distiller.load_checkpoint(args.resume)

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
        stages = method_cfg["stages"]
        t_steps, s_steps = stages[args.stage]
        run_name = f"pd_stage{args.stage}_{t_steps}to{s_steps}_{datetime.now():%m%d_%H%M}"
    else:
        total_steps = method_cfg.get("total_steps", global_total_steps)
        run_name = f"{args.method}_{total_steps}steps_{datetime.now():%m%d_%H%M}"

    distiller.set_run_name(run_name)
    distiller.train(dataloader, total_steps=total_steps, resume_step=resume_step)

    # For PD, save merged model after each stage for next stage handoff
    if args.method == "pd" and distiller.is_main:
        distiller.save_merged_model(args.stage)

    # Cleanup DDP
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
