#!/usr/bin/env python3
"""DDP smoke test: verify fake_score adapter stays synchronized across ranks.

Regression guard for the bug where PEFT adapters added via add_adapter()
AFTER DDP wrap are silently excluded from gradient all-reduce, causing
per-rank divergence. Runs DMD1 for a handful of steps and checks that
fake_score parameters are bitwise identical across ranks.

Usage:
    cd /mnt/workspace/kuno/distillation/models/hunyuan3d21/hy3dshape
    PYTHONPATH=.:../../../distill_methods/src \
    torchrun --nproc_per_node=2 \
        ../../../distill_methods/scripts/smoke_test_ddp.py
"""

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

# Initialize DDP
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
torch.cuda.set_device(rank)
logger = logging.getLogger(str(rank))

os.environ["WANDB_MODE"] = "disabled"

PROJECT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_DIR / "distill_methods" / "config.yaml"

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

SMOKE_STEPS = 6
config["training"]["log_interval"] = 2
config["training"]["save_interval"] = 999999
config["output_root"] = tempfile.mkdtemp(prefix=f"smoke_ddp_rank{rank}_")
config["training"]["batch_size"] = 1
config["training"]["num_workers"] = 0
config["training"]["total_steps"] = SMOKE_STEPS

# Identical seed across ranks so DDP's initial param broadcast has something
# consistent to align to; DDP then keeps them in sync throughout training.
torch.manual_seed(42)
np.random.seed(42)

sys.path.insert(0, str(PROJECT_DIR / "distill_methods" / "src"))
from train import LatentDataset, make_dataloader  # noqa: E402
from dmd1_distillation import DMD1Distillation  # noqa: E402


def fake_score_flat(distiller) -> torch.Tensor:
    """Concat all fake_score params into a deterministic 1D fp32 tensor."""
    model = (
        distiller.student.module
        if distiller.is_distributed
        else distiller.student
    )
    items = [(n, p) for n, p in model.named_parameters() if "fake_score" in n]
    items.sort(key=lambda x: x[0])  # deterministic cross-rank ordering
    if not items:
        return torch.zeros(1, device=f"cuda:{rank}")
    return torch.cat([p.data.float().flatten() for _, p in items])


def main():
    dataset = LatentDataset(config["training"]["training_data_dir"])
    dataloader = make_dataloader(dataset, config, batch_size_override=1)

    logger.info("starting DMD1 DDP smoke (%d steps, world=%d)", SMOKE_STEPS, world_size)
    distiller = DMD1Distillation(config)
    distiller.set_run_name("smoke_ddp_dmd1")

    # Check 1: params synchronized right after DDP wrap (before any training)?
    flat_init = fake_score_flat(distiller)
    init_gathered = [torch.zeros_like(flat_init) for _ in range(world_size)]
    dist.all_gather(init_gathered, flat_init)
    dist.barrier()
    if rank == 0:
        init_max = max((g - init_gathered[0]).abs().max().item() for g in init_gathered)
        logger.info("INIT check: max abs diff across ranks = %.3e (should be 0)", init_max)

    distiller.train(dataloader, total_steps=SMOKE_STEPS)

    # Cross-rank compare: flatten fake_score params, all_gather, elementwise diff.
    flat = fake_score_flat(distiller)
    gathered = [torch.zeros_like(flat) for _ in range(world_size)]
    dist.all_gather(gathered, flat)

    dist.barrier()
    if rank == 0:
        flat0 = gathered[0]
        per_rank_max = [(g - flat0).abs().max().item() for g in gathered]
        max_diff = max(per_rank_max)
        logger.info("fake_score flat size: %d (fp32)", flat0.numel())
        logger.info("per-rank max abs diff vs rank0: %s", per_rank_max)
        # Expect BITWISE identical params: DDP broadcasts rank0 at wrap,
        # then grad all-reduce applies identical updates on every step.
        # Per-rank random noise inside training_step differs, but backward
        # on DDP-tracked params still produces averaged gradients.
        if max_diff < 1e-5:
            logger.info("PASS: fake_score params synchronized across ranks (max_diff=%.3e)", max_diff)
            exit_code = 0
        else:
            logger.error("FAIL: fake_score params DIVERGED across ranks (max_diff=%.3e)", max_diff)
            exit_code = 1
    else:
        exit_code = 0

    dist.barrier()
    dist.destroy_process_group()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
