#!/usr/bin/env python3
"""Merge a LoRA checkpoint into the pretrained DiT and save the full state dict.

Used to create stage_N_merged/model.pt when training crashed before
save_merged_model() could run. The merged model is needed as the base
for the next PD stage.

Optionally runs a few stage 1 training steps to verify the merge is valid.

Usage:
    cd /mnt/workspace/kuno/distillation/models/hunyuan3d21/hy3dshape
    PYTHONPATH=.:../../../distill_methods/src CUDA_VISIBLE_DEVICES=0 \
        ../../../envs/hunyuan3d-venv/bin/python -u \
        ../../../distill_methods/scripts/merge_lora_checkpoint.py \
        --config ../../../distill_methods/config.yaml \
        --lora-path ../../../results/distill_methods/checkpoints/pd/step_4000 \
        --stage 0 \
        --verify-stage1
"""

import argparse
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
logger = logging.getLogger("merge_lora")


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA into pretrained DiT")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--lora-path", required=True, help="LoRA checkpoint directory (e.g. step_4000)")
    parser.add_argument("--stage", type=int, required=True, help="Stage index being merged (0-based)")
    parser.add_argument("--verify-stage1", action="store_true", help="Run a few stage 1 steps to verify")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda")
    output_root = config["output_root"]
    ckpt_dir = os.path.join(output_root, "checkpoints", "pd")

    # --- Load pipeline (gets pretrained DiT) ---
    logger.info("Loading pipeline (this may take ~5 min on first load) ...")
    from base_distiller import _load_pipeline
    dit, pipeline = _load_pipeline(config["training"]["model"]["pretrained"])
    dit = dit.to(device)
    logger.info("Pipeline loaded. DiT: %d params", sum(p.numel() for p in dit.parameters()))

    # --- Apply LoRA from checkpoint ---
    logger.info("Loading LoRA adapter from %s ...", args.lora_path)
    from peft import PeftModel
    dit_lora = PeftModel.from_pretrained(dit, args.lora_path, is_trainable=True)
    trainable = sum(p.numel() for p in dit_lora.parameters() if p.requires_grad)
    logger.info("LoRA loaded: %d trainable params", trainable)

    # Sanity: adapter weights should be non-zero
    adapter_norms = []
    for name, param in dit_lora.named_parameters():
        if "lora" in name and param.requires_grad:
            adapter_norms.append(param.data.norm().item())
    avg_norm = sum(adapter_norms) / len(adapter_norms) if adapter_norms else 0
    logger.info("Adapter avg norm: %.6f", avg_norm)
    assert avg_norm > 0, "Adapter weights are all zeros — wrong checkpoint?"

    # --- Merge and save ---
    logger.info("Merging LoRA into base ...")
    merged = dit_lora.merge_and_unload()
    merged_keys = list(merged.state_dict().keys())
    logger.info("Merged model: %d keys", len(merged_keys))

    save_dir = os.path.join(ckpt_dir, f"stage_{args.stage}_merged")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "model.pt")
    torch.save(merged.state_dict(), save_path)
    size_gb = os.path.getsize(save_path) / 1e9
    logger.info("Saved merged model: %s (%.1f GB, %d keys)", save_path, size_gb, len(merged_keys))

    # Verify no NaN/Inf
    for name, param in merged.named_parameters():
        assert not torch.isnan(param).any(), f"NaN in {name}"
        assert not torch.isinf(param).any(), f"Inf in {name}"
    logger.info("Merged model passes NaN/Inf check.")

    # --- Optional: verify stage 1 startup ---
    if args.verify_stage1:
        logger.info("=== Verifying stage 1 startup ===")
        next_stage = args.stage + 1
        stages = config["training"]["methods"]["pd"]["stages"]
        if next_stage >= len(stages):
            logger.info("No next stage to verify (stage %d was the last).", args.stage)
            return

        teacher_steps, student_steps = stages[next_stage]
        h = 1.0 / teacher_steps
        logger.info("Stage %d: %d-step teacher -> %d-step student (h=%.4f)",
                     next_stage, teacher_steps, student_steps, h)

        # Load merged weights into teacher
        state_dict = merged.state_dict()
        teacher = merged  # reuse the merged model as teacher
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        # Build fresh student with LoRA on top of merged base
        from hy3dshape.models.denoisers.hunyuandit import HunYuanDiTPlain
        import copy
        student_base = copy.deepcopy(merged)
        for p in student_base.parameters():
            p.requires_grad_(True)

        from peft import LoraConfig, get_peft_model
        lora_cfg = config["training"]["model"]["lora"]
        peft_config = LoraConfig(
            r=lora_cfg["rank"],
            lora_alpha=lora_cfg["alpha"],
            target_modules=lora_cfg["target_modules"],
        )
        student = get_peft_model(student_base, peft_config)
        student_trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
        logger.info("Student: %d trainable params", student_trainable)

        optimizer = torch.optim.AdamW(
            [p for p in student.parameters() if p.requires_grad],
            lr=config["training"]["methods"]["pd"]["lr"],
            weight_decay=0.01,
        )

        # Synthetic forward pass
        B = 1
        x_data = torch.randn(B, 4096, 64, device=device, dtype=torch.float32)
        noise = torch.randn_like(x_data)
        contexts = {"main": torch.randn(B, 1370, 1024, device=device, dtype=torch.float32)}

        num_verify_steps = 3
        for step_i in range(num_verify_steps):
            t_idx = torch.randint(0, student_steps, (B,), device=device)
            t = (t_idx.float() * 2.0 * h).clamp(0.0, 1.0 - 2.0 * h)

            t_view = t.view(-1, 1, 1)
            x_t = t_view * x_data + (1 - t_view) * noise

            # Teacher 2-step
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                v1 = teacher(x_t, t, contexts=contexts)
                x_mid = x_t + h * v1
                t_mid = t + h
                v2 = teacher(x_mid, t_mid, contexts=contexts)
                x_tgt = x_mid + h * v2

            # Student 1-step
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v_s = student(x_t, t, contexts=contexts)
            x_pred = x_t + 2.0 * h * v_s

            loss = torch.nn.functional.mse_loss(x_pred, x_tgt.detach())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in student.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            optimizer.zero_grad()

            logger.info("  verify step %d: loss=%.6f (finite=%s)",
                        step_i, loss.item(), torch.isfinite(loss).item())
            assert torch.isfinite(loss), f"Non-finite loss at verify step {step_i}!"

        logger.info("=== Stage 1 verification PASSED ===")

    logger.info("=== DONE ===")


if __name__ == "__main__":
    main()
