#!/usr/bin/env python3
"""Test PD stage handoff: merge step_4000 LoRA, save, reload as new base.

Verifies:
1. LoRA merge works
2. Merged model can be saved/loaded
3. Fresh LoRA can be applied on top
4. Stage 1 schedule (25->12) is set correctly

Usage:
    cd /mnt/workspace/kuno/distillation/models/hunyuan3d21/hy3dshape
    PYTHONPATH=.:../../../distill_methods/src CUDA_VISIBLE_DEVICES=0 \
        ../../../envs/hunyuan3d-venv/bin/python -u \
        ../../../distill_methods/scripts/test_stage_handoff.py
"""

import logging
import os
import sys
from pathlib import Path

import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_stage_handoff")

PROJECT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_DIR / "distill_methods" / "config.yaml"
STEP_4000_PATH = PROJECT_DIR / "results" / "distill_methods" / "checkpoints" / "pd" / "step_4000"
MERGED_DIR = PROJECT_DIR / "results" / "distill_methods" / "checkpoints" / "pd" / "stage_0_merged"

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

# Reduce workers to avoid issues
config["training"]["num_workers"] = 0

logger.info("=== Step 1: Build PD distiller ===")
sys.path.insert(0, str(PROJECT_DIR / "distill_methods" / "src"))
from progressive_distillation import ProgressiveDistillation

distiller = ProgressiveDistillation(config)
distiller.set_stage(0)

logger.info("=== Step 2: Load step_4000 LoRA checkpoint ===")
distiller.load_checkpoint(str(STEP_4000_PATH))
logger.info("Loaded step_4000 LoRA.")

# Verify student has LoRA
model = distiller.student.module if distiller.is_distributed else distiller.student
from peft import PeftModel
assert isinstance(model, PeftModel), "Student should be a PeftModel"
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info("Student trainable params: %d", trainable)

logger.info("=== Step 3: Save merged model (stage 0) ===")
distiller.save_merged_model(0)
merged_path = os.path.join(str(MERGED_DIR), "model.pt")
assert os.path.exists(merged_path), f"Merged model not found at {merged_path}"
size_mb = os.path.getsize(merged_path) / 1e6
logger.info("Merged model saved: %s (%.0f MB)", merged_path, size_mb)

logger.info("=== Step 4: Load merged base for stage 1 ===")
distiller.load_merged_base(merged_path)

# Verify fresh LoRA on student
model = distiller.student.module if distiller.is_distributed else distiller.student
assert isinstance(model, PeftModel), "Student should have fresh LoRA"
trainable_new = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info("Fresh LoRA trainable params: %d", trainable_new)
assert trainable_new == trainable, "Trainable param count should match"

logger.info("=== Step 5: Set stage 1 and verify schedule ===")
distiller.set_stage(1)
assert distiller.teacher_steps == 25, f"Expected teacher_steps=25, got {distiller.teacher_steps}"
assert distiller.student_steps == 12, f"Expected student_steps=12, got {distiller.student_steps}"
assert abs(distiller.h - 1.0/25) < 1e-6, f"Expected h=0.04, got {distiller.h}"
logger.info("Stage 1 schedule: %d -> %d steps, h=%.4f", distiller.teacher_steps, distiller.student_steps, distiller.h)

logger.info("=== Step 6: Quick forward pass test ===")
# Create dummy batch
x_data = torch.randn(1, 4096, 64, device=distiller.device)
image_cond = torch.randn(1, 1370, 1024, device=distiller.device)
batch = {"latent": x_data, "image_cond": image_cond}

loss_dict = distiller.training_step(batch, step=0)
logger.info("Training step loss: %.6f", loss_dict["loss"].item())
assert not torch.isnan(loss_dict["loss"]), "Loss is NaN!"
assert not torch.isinf(loss_dict["loss"]), "Loss is Inf!"

logger.info("=== ALL TESTS PASSED ===")
