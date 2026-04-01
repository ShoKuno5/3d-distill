#!/usr/bin/env python3
"""Lightweight stage handoff test — no full pipeline load.

Tests LoRA merge -> save -> reload -> fresh LoRA cycle
using only the DiT model (no VAE, no conditioner, no CPFS wait).

Also tests the load_checkpoint deepcopy path to ensure adapter
weights survive round-tripping through a PeftModel.

Usage:
    cd /mnt/workspace/kuno/distillation/models/hunyuan3d21/hy3dshape
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 ../../../envs/hunyuan3d-venv/bin/python -u \
        ../../../distill_methods/scripts/test_stage_handoff_lite.py
"""

import copy
import logging
import os
import tempfile

import torch
import yaml
from peft import LoraConfig, PeftModel, get_peft_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("handoff_lite")

CKPT_DIR = "/mnt/workspace/kuno/distillation/results/distill_methods/checkpoints/pd"
STEP_4000 = os.path.join(CKPT_DIR, "step_4000")

LORA_CONFIG = LoraConfig(
    r=64,
    lora_alpha=64,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
)


def build_dit():
    """Build a DiT from config with random weights (no checkpoint load)."""
    config_path = os.path.expanduser(
        "~/.cache/hy3dgen/tencent/Hunyuan3D-2.1/hunyuan3d-dit-v2-1/config.yaml"
    )
    with open(config_path) as f:
        model_config = yaml.safe_load(f)

    from hy3dshape.models.denoisers.hunyuandit import HunYuanDiTPlain

    dit = HunYuanDiTPlain(**model_config["model"]["params"])
    return dit.cuda().half()


# --- Step 1: Build DiT ---
logger.info("=== Step 1: Build DiT from config (random weights) ===")
dit = build_dit()
total_params = sum(p.numel() for p in dit.parameters())
logger.info("DiT built: %d params (%.1fB)", total_params, total_params / 1e9)

# --- Step 2: Apply LoRA ---
logger.info("=== Step 2: Apply LoRA (rank=64) ===")
dit_lora = get_peft_model(dit, LORA_CONFIG)
trainable = sum(p.numel() for p in dit_lora.parameters() if p.requires_grad)
logger.info("LoRA applied: %d trainable params (%.2f%%)", trainable, 100 * trainable / total_params)

# --- Step 3: Load step_4000 adapter onto fresh base ---
logger.info("=== Step 3: Load step_4000 adapter onto fresh base ===")
dit_fresh = build_dit()
dit_loaded = PeftModel.from_pretrained(dit_fresh, STEP_4000, is_trainable=True)
trainable_loaded = sum(p.numel() for p in dit_loaded.parameters() if p.requires_grad)
logger.info("Loaded adapter: %d trainable params", trainable_loaded)

adapter_norms = []
for name, param in dit_loaded.named_parameters():
    if "lora" in name and param.requires_grad:
        adapter_norms.append(param.data.norm().item())
avg_norm = sum(adapter_norms) / len(adapter_norms) if adapter_norms else 0
logger.info("Adapter weight avg norm: %.6f (should be > 0)", avg_norm)
assert avg_norm > 0, "Adapter weights are all zeros!"

# --- Step 3b: Test load_checkpoint deepcopy path ---
logger.info("=== Step 3b: Test load_checkpoint deepcopy path ===")
# Simulate: student is already a PeftModel, then we load a new adapter
dit_base2 = build_dit()
existing_peft = get_peft_model(dit_base2, LORA_CONFIG)  # current student (PeftModel)

# This is what the buggy code did — extract base_model.model without deepcopy
base_no_copy = existing_peft.base_model.model
has_peft_residue = hasattr(base_no_copy, "peft_config") or hasattr(base_no_copy, "_hf_peft_config_loaded")
logger.info("base_model.model has peft residue (no deepcopy): %s", has_peft_residue)

# Fixed path: deepcopy severs peft state
base_clean = copy.deepcopy(existing_peft.base_model.model)
loaded_from_clean = PeftModel.from_pretrained(base_clean, STEP_4000, is_trainable=True)
trainable_clean = sum(p.numel() for p in loaded_from_clean.parameters() if p.requires_grad)
logger.info("Deepcopy + from_pretrained: %d trainable params (should be %d)", trainable_clean, trainable)
assert trainable_clean == trainable, f"Deepcopy path: expected {trainable}, got {trainable_clean} trainable params"

# Verify loaded adapter norms match
clean_norms = []
for name, param in loaded_from_clean.named_parameters():
    if "lora" in name and param.requires_grad:
        clean_norms.append(param.data.norm().item())
avg_clean = sum(clean_norms) / len(clean_norms) if clean_norms else 0
logger.info("Deepcopy adapter avg norm: %.6f (should match %.6f)", avg_clean, avg_norm)
assert abs(avg_clean - avg_norm) < 1e-4, "Deepcopy path adapter norms differ!"

del existing_peft, base_no_copy, base_clean, loaded_from_clean, dit_base2
torch.cuda.empty_cache()

# --- Step 4: Merge LoRA into base ---
logger.info("=== Step 4: Merge LoRA into base ===")
merged = dit_loaded.merge_and_unload()
merged_params = sum(p.numel() for p in merged.parameters())
logger.info("Merged model: %d params", merged_params)
assert merged_params == total_params, "Param count mismatch after merge"
assert not any(torch.isnan(p).any() for p in merged.parameters()), "NaN in merged weights"
assert not any(torch.isinf(p).any() for p in merged.parameters()), "Inf in merged weights"

# --- Step 5: Save merged state dict to tmpdir ---
logger.info("=== Step 5: Save merged model (tmpdir) ===")
with tempfile.TemporaryDirectory() as tmpdir:
    save_path = os.path.join(tmpdir, "model.pt")
    torch.save(merged.state_dict(), save_path)
    size_mb = os.path.getsize(save_path) / 1e6
    logger.info("Saved: %s (%.0f MB)", save_path, size_mb)

    # --- Step 6: Reload merged as teacher + fresh LoRA student ---
    logger.info("=== Step 6: Reload merged as teacher + fresh LoRA student ===")
    teacher = build_dit()
    state_dict = torch.load(save_path, map_location="cuda")
    teacher.load_state_dict(state_dict)
    teacher.eval()
    logger.info("Teacher loaded from merged checkpoint")

    student_base = build_dit()
    student_base.load_state_dict(state_dict)
    student = get_peft_model(student_base, LORA_CONFIG)
    trainable_fresh = sum(p.numel() for p in student.parameters() if p.requires_grad)
    logger.info("Student with fresh LoRA: %d trainable params", trainable_fresh)
    assert trainable_fresh == trainable, "Fresh LoRA param count mismatch"

    fresh_norms = []
    for name, param in student.named_parameters():
        if "lora" in name and param.requires_grad:
            fresh_norms.append(param.data.norm().item())
    avg_fresh = sum(fresh_norms) / len(fresh_norms) if fresh_norms else 0
    logger.info("Fresh LoRA avg norm: %.6f (should be ~0)", avg_fresh)

    # --- Step 7: Forward pass test ---
    logger.info("=== Step 7: Forward pass with stage 1 schedule ===")
    B = 1
    x_t = torch.randn(B, 4096, 64, device="cuda", dtype=torch.half)
    t = torch.tensor([0.04], device="cuda")  # h = 1/25 for stage 1
    contexts = {"main": torch.randn(B, 1370, 1024, device="cuda", dtype=torch.half)}

    with torch.no_grad():
        v_teacher = teacher(x_t, t, contexts=contexts)
    logger.info("Teacher forward: output shape %s, norm %.4f", v_teacher.shape, v_teacher.norm().item())

    v_student = student(x_t, t, contexts=contexts)
    logger.info("Student forward: output shape %s, norm %.4f", v_student.shape, v_student.norm().item())

    assert v_teacher.shape == v_student.shape, "Shape mismatch"
    assert not torch.isnan(v_teacher).any(), "Teacher NaN"
    assert not torch.isnan(v_student).any(), "Student NaN"

    mse = torch.nn.functional.mse_loss(v_student, v_teacher)
    logger.info("MSE(teacher, student): %.6f", mse.item())

logger.info("=== ALL TESTS PASSED ===")
