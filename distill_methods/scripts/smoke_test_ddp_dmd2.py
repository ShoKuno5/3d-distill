#!/usr/bin/env python3
"""DDP smoke test for DMD2: discriminator + replay buffer cross-rank sync.

Extends scripts/smoke_test_ddp.py (which checks fake_score adapter sync).
DMD2 adds two new structures not covered there:

1. ``self.discriminator`` (FeatureDiscriminator) is NOT wrapped in DDP — see
   dmd2_distillation.py:102. Its parameters can diverge across ranks if each
   rank trains the discriminator on its own local batch.

2. ``ReplayBuffer.sample`` calls torch.randint without a seed
   (dmd2_distillation.py:64), so each rank samples different indices from
   the replay buffer for "real" batches.

This script runs DMD2 for a handful of steps with replay_warmup overridden
to a tiny value (so ``_train_discriminator`` actually fires), then for each
of the two structures runs ``dist.all_gather`` + element-wise diff against
rank 0. Two PASS/FAIL lines are emitted at the end.

Usage:
    cd <repo>/models/hunyuan3d21/hy3dshape
    PYTHONPATH=.:<repo>/distill_methods/src \\
      CUDA_VISIBLE_DEVICES=0,1 \\
      torchrun --nproc_per_node=2 \\
        <repo>/distill_methods/scripts/smoke_test_ddp_dmd2.py \\
        --config <repo>/distill_methods/configs/config_525_hssd_tsubame.yaml
"""

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] rank=%(name)s: %(message)s",
)

dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
# In multi-node DDP, dist.get_rank() is the *global* rank but each node
# only exposes its local GPUs (cuda:0..N-1). Use LOCAL_RANK from torchrun.
local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")
logger = logging.getLogger(str(rank))

os.environ["WANDB_MODE"] = "disabled"

PROJECT_DIR = Path(__file__).resolve().parents[2]


def discriminator_flat(distiller) -> torch.Tensor:
    """Concat all discriminator params into a deterministic 1D fp32 tensor."""
    items = sorted(distiller.discriminator.named_parameters(), key=lambda x: x[0])
    if not items:
        return torch.zeros(1, device=device)
    return torch.cat([p.data.float().flatten() for _, p in items])


def replay_indices_sample(distiller, batch_size: int) -> torch.Tensor:
    """Reproduce ReplayBuffer.sample's torch.randint call (no seed)."""
    n = len(distiller.replay_buffer.buffer)
    return torch.randint(0, n, (batch_size,), device=device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    SMOKE_STEPS = args.steps
    config["training"]["log_interval"] = 2
    config["training"]["save_interval"] = 999999
    config["output_root"] = tempfile.mkdtemp(prefix=f"smoke_ddp_dmd2_rank{rank}_")
    config["training"]["batch_size"] = 1
    config["training"]["num_workers"] = 0
    config["training"]["total_steps"] = SMOKE_STEPS

    # Reduce replay_warmup so _train_discriminator fires within SMOKE_STEPS
    config["training"]["methods"]["dmd2"]["replay_warmup"] = 4
    config["training"]["methods"]["dmd2"]["replay_buffer_size"] = 64

    # Identical seed across ranks so DDP's initial param broadcast can align.
    torch.manual_seed(42)
    np.random.seed(42)

    sys.path.insert(0, str(PROJECT_DIR / "distill_methods" / "src"))
    from train import LatentDataset, make_dataloader  # noqa: E402
    from dmd2_distillation import DMD2Distillation  # noqa: E402

    dataset = LatentDataset(config["training"]["training_data_dir"])
    dataloader = make_dataloader(dataset, config, batch_size_override=1)

    logger.info(
        "starting DMD2 DDP smoke (%d steps, world=%d, replay_warmup=4)",
        SMOKE_STEPS, world_size,
    )
    distiller = DMD2Distillation(config)
    distiller.set_run_name("smoke_ddp_dmd2")

    # --- Check 0: discriminator params identical right after init (same seed) ---
    flat_init = discriminator_flat(distiller)
    init_gathered = [torch.zeros_like(flat_init) for _ in range(world_size)]
    dist.all_gather(init_gathered, flat_init)
    dist.barrier()
    if rank == 0:
        init_max = max((g - init_gathered[0]).abs().max().item() for g in init_gathered)
        logger.info(
            "INIT: discriminator max abs diff across ranks = %.3e (expect 0 since manual_seed identical)",
            init_max,
        )

    # --- Run a few steps so _train_discriminator fires ---
    distiller.train(dataloader, total_steps=SMOKE_STEPS)

    # --- Check 1: discriminator params after training ---
    flat = discriminator_flat(distiller)
    gathered = [torch.zeros_like(flat) for _ in range(world_size)]
    dist.all_gather(gathered, flat)
    dist.barrier()

    # --- Check 2: ReplayBuffer.sample indices match across ranks? ---
    # Pre-fill more if needed (training may have already added enough)
    if len(distiller.replay_buffer.buffer) < 4:
        # Should have at least replay_warmup items; if not, fill dummies
        for _ in range(4):
            distiller.replay_buffer.buffer.append(
                (torch.zeros(8, 4), torch.zeros(8, 4))
            )
    sampled_idx = replay_indices_sample(distiller, batch_size=8)
    idx_gathered = [torch.zeros_like(sampled_idx) for _ in range(world_size)]
    dist.all_gather(idx_gathered, sampled_idx)
    dist.barrier()

    exit_code = 0
    if rank == 0:
        # --- Discriminator divergence summary ---
        flat0 = gathered[0]
        per_rank_max = [(g - flat0).abs().max().item() for g in gathered]
        disc_diff = max(per_rank_max)
        logger.info("discriminator flat size: %d (fp32)", flat0.numel())
        logger.info("per-rank max abs diff vs rank0: %s", per_rank_max)
        if disc_diff < 1e-5:
            logger.info("PASS (H1): discriminator params SYNCHRONIZED across ranks (max=%.3e)", disc_diff)
        else:
            logger.error(
                "FAIL (H1): discriminator params DIVERGED across ranks (max=%.3e). "
                "Likely cause: discriminator is not DDP-wrapped (dmd2_distillation.py:102).",
                disc_diff,
            )
            exit_code = 1

        # --- Replay buffer indices summary ---
        idx0 = idx_gathered[0]
        per_rank_mismatch = [(g != idx0).sum().item() for g in idx_gathered]
        idx_total_diff = max(per_rank_mismatch)
        logger.info("replay sample indices (rank0): %s", idx0.tolist())
        for r, g in enumerate(idx_gathered):
            if r > 0:
                logger.info("replay sample indices (rank%d): %s", r, g.tolist())
        if idx_total_diff == 0:
            logger.info("PASS (H2): ReplayBuffer.sample indices ALIGNED across ranks")
        else:
            logger.error(
                "FAIL (H2): ReplayBuffer.sample indices DIFFER across ranks "
                "(%d/%d positions). Likely cause: torch.randint without seed "
                "(dmd2_distillation.py:64).",
                idx_total_diff, idx0.numel(),
            )
            exit_code = 1

        if exit_code == 0:
            logger.info("All checks PASS — no DDP issues detected in DMD2 discriminator/replay paths.")
        else:
            logger.error("One or more DDP checks FAILED. See per-rank diff above.")

    dist.barrier()
    dist.destroy_process_group()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
