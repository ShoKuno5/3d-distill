#!/usr/bin/env python3
"""Smoke test for Progressive Distillation: 20 steps, minimal config.

Usage:
    cd /mnt/workspace/kuno/distillation/models/hunyuan3d21/hy3dshape
    PYTHONPATH=.:../../../distill_methods/src \
    CUDA_VISIBLE_DEVICES=0 ../../../envs/hunyuan3d-venv/bin/python \
        ../../../distill_methods/scripts/smoke_test_pd.py
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
logger = logging.getLogger("smoke_test_pd")

# Paths
PROJECT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_DIR / "distill_methods" / "config.yaml"

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

# Override for smoke test
config["training"]["methods"]["pd"]["steps_per_stage"] = 20
config["training"]["log_interval"] = 1
config["training"]["save_interval"] = 999999  # don't save during smoke test
config["training"]["batch_size"] = 1  # small batch for 3 samples
config["training"]["num_workers"] = 0  # avoid multiprocessing issues

torch.manual_seed(42)
np.random.seed(42)

# Dataset
sys.path.insert(0, str(PROJECT_DIR / "distill_methods" / "src"))
from train import LatentDataset, make_dataloader

dataset = LatentDataset(config["training"]["training_data_dir"])
dataloader = make_dataloader(dataset, config, batch_size_override=1)

logger.info("Dataset: %d samples", len(dataset))

# Build distiller
t0 = time.time()
from progressive_distillation import ProgressiveDistillation

distiller = ProgressiveDistillation(config)
logger.info("Model loaded in %.1fs", time.time() - t0)

# GPU memory after model load
if torch.cuda.is_available():
    mem_alloc = torch.cuda.memory_allocated() / 1e9
    mem_reserved = torch.cuda.memory_reserved() / 1e9
    logger.info("GPU memory after load: %.1f GB allocated, %.1f GB reserved", mem_alloc, mem_reserved)

# Set stage 0
distiller.set_stage(0)

# Run 20 training steps
logger.info("=== Starting 20-step smoke test ===")
t_start = time.time()
distiller.train(dataloader, total_steps=20)
elapsed = time.time() - t_start

logger.info("=== Smoke test complete ===")
logger.info("20 steps in %.1fs (%.2f s/step)", elapsed, elapsed / 20)

if torch.cuda.is_available():
    mem_alloc = torch.cuda.memory_allocated() / 1e9
    mem_reserved = torch.cuda.memory_reserved() / 1e9
    mem_peak = torch.cuda.max_memory_allocated() / 1e9
    logger.info("GPU memory after training: %.1f GB allocated, %.1f GB reserved, %.1f GB peak", mem_alloc, mem_reserved, mem_peak)

logger.info("PASSED")
