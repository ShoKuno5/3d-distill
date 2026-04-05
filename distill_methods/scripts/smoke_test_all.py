#!/usr/bin/env python3
"""Smoke test all 4 distillation methods: 10 steps each, verify loss is finite.

Usage:
    cd /mnt/workspace/kuno/distillation/models/hunyuan3d21/hy3dshape
    PYTHONPATH=.:../../../distill_methods/src \
    CUDA_VISIBLE_DEVICES=0 ../../../envs/hunyuan3d-venv/bin/python \
        ../../../distill_methods/scripts/smoke_test_all.py
"""

import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("smoke_test_all")

# Wandb API key
os.environ.setdefault("WANDB_API_KEY", "offline")
os.environ["WANDB_MODE"] = "disabled"

PROJECT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_DIR / "distill_methods" / "config.yaml"

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

# Minimal overrides for smoke test
SMOKE_STEPS = 10
config["training"]["log_interval"] = 5
config["training"]["save_interval"] = 999999
# Use a temp dir for checkpoints to avoid permission issues
import tempfile
config["output_root"] = tempfile.mkdtemp(prefix="smoke_test_")
config["training"]["batch_size"] = 1
config["training"]["num_workers"] = 0
config["training"]["methods"]["pd"]["steps_per_stage"] = SMOKE_STEPS
config["training"]["total_steps"] = SMOKE_STEPS

torch.manual_seed(42)
np.random.seed(42)

sys.path.insert(0, str(PROJECT_DIR / "distill_methods" / "src"))
from train import LatentDataset, make_dataloader

dataset = LatentDataset(config["training"]["training_data_dir"])
dataloader = make_dataloader(dataset, config, batch_size_override=1)
logger.info("Dataset: %d samples", len(dataset))

results = {}


def run_method(name, cls, setup_fn=None):
    """Run a single method for SMOKE_STEPS steps and check loss."""
    logger.info("=" * 60)
    logger.info("Testing: %s", name)
    logger.info("=" * 60)
    torch.manual_seed(42)

    t0 = time.time()
    distiller = cls(config)
    load_time = time.time() - t0
    logger.info("%s: model loaded in %.1fs", name, load_time)

    if setup_fn:
        setup_fn(distiller)

    t_start = time.time()
    distiller.train(dataloader, total_steps=SMOKE_STEPS)
    elapsed = time.time() - t_start

    # Check that training produced finite losses by running one more step
    batch = next(iter(dataloader))
    batch = {k: v.to(distiller.device) if isinstance(v, torch.Tensor) else v
             for k, v in batch.items()}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model = distiller.student.module if distiller.is_distributed else distiller.student
        model.eval()
        x = batch["latent"]
        t_test = torch.tensor([0.5], device=distiller.device)
        contexts = {"main": batch["image_cond"]}
        v = model(x, t_test.expand(x.shape[0]), contexts=contexts)
        assert v.isfinite().all(), f"{name}: non-finite output after training!"
        model.train()

    mem_peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    torch.cuda.reset_peak_memory_stats()

    results[name] = {
        "time": elapsed,
        "time_per_step": elapsed / SMOKE_STEPS,
        "peak_memory_gb": mem_peak,
        "status": "PASSED",
    }
    logger.info(
        "%s: %d steps in %.1fs (%.2f s/step), peak %.1f GB — PASSED",
        name, SMOKE_STEPS, elapsed, elapsed / SMOKE_STEPS, mem_peak,
    )

    # Free GPU memory
    del distiller
    torch.cuda.empty_cache()


# --- Import all methods ---
from progressive_distillation import ProgressiveDistillation
from consistency_distillation import ConsistencyDistillation
from dmd1_distillation import DMD1Distillation
from dmd2_distillation import DMD2Distillation

# --- PD ---
run_method("PD (stage 0)", ProgressiveDistillation,
           setup_fn=lambda d: d.set_stage(0))

# --- CD ---
run_method("CD", ConsistencyDistillation)

# --- DMD1 ---
run_method("DMD1", DMD1Distillation)

# --- DMD2 ---
run_method("DMD2", DMD2Distillation)

# --- Summary ---
logger.info("=" * 60)
logger.info("SUMMARY")
logger.info("=" * 60)
all_passed = True
for name, r in results.items():
    logger.info("  %-20s %s  (%.2f s/step, %.1f GB peak)",
                name, r["status"], r["time_per_step"], r["peak_memory_gb"])
    if r["status"] != "PASSED":
        all_passed = False

if all_passed:
    logger.info("ALL METHODS PASSED")
else:
    logger.error("SOME METHODS FAILED")
    sys.exit(1)
