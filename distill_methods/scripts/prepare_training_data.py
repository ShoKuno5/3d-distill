#!/usr/bin/env python3
"""Prepare pre-encoded training data for distillation.

For each sample: run the image through the conditioner to get image_cond,
then run the teacher DiT (50-step ODE with CFG) to generate a latent.
Saves (latent, image_cond) pairs as NPZ files.

Usage:
    cd /mnt/workspace/kuno/distillation/models/hunyuan3d21/hy3dshape
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 ../../../envs/hunyuan3d-venv/bin/python \
        ../../../distill_methods/scripts/prepare_training_data.py \
        --config ../../../distill_methods/config.yaml
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_DIR / "pipeline"))
sys.path.insert(0, str(PROJECT_DIR / "models" / "hunyuan3d21" / "hy3dshape"))

from src.utils.inference_config import load_and_filter_samples


def teacher_ode_solve(model, noise, contexts, num_steps=50, guidance_scale=7.5):
    """Run the teacher's Euler ODE from t=0 (noise) to t=1 (data).

    ICPlan convention: x_t = t * x_data + (1-t) * noise, v = x_data - noise.
    Euler step: x_{t+dt} = x_t + dt * v(x_t, t).
    """
    dt = 1.0 / num_steps
    x_t = noise.clone()

    for i in range(num_steps):
        t = torch.full((noise.shape[0],), i * dt, device=noise.device)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            v_cond = model(x_t, t, contexts=contexts)
            if guidance_scale > 1.0:
                null_ctx = {k: torch.zeros_like(v) for k, v in contexts.items()}
                v_uncond = model(x_t, t, contexts=null_ctx)
                v = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                v = v_cond

        x_t = x_t + dt * v

    return x_t


def main():
    parser = argparse.ArgumentParser(description="Prepare training data for distillation")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    samples = load_and_filter_samples(config, max_samples_override=args.max_samples)
    out_dir = config["training"]["training_data_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # Check existing
    existing = set()
    for s in samples:
        npz_path = os.path.join(out_dir, f"{s.object_id}.npz")
        if os.path.exists(npz_path):
            existing.add(s.object_id)

    missing = [s for s in samples if s.object_id not in existing]
    print(f"Training data: {len(existing)}/{len(samples)} exist, {len(missing)} to encode")

    if not missing:
        print("All training data ready.")
        return

    # Load pipeline
    print("Loading Hunyuan3D-2.1 pipeline ...")
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    pretrained = config["training"]["model"]["pretrained"]
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        pretrained, use_safetensors=False,
    )
    teacher = pipeline.model
    cond_model = pipeline.conditioner
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move teacher + conditioner to GPU (VAE not needed)
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    cond_model.to(device)

    # Free VAE from GPU (it was auto-loaded by from_pretrained)
    pipeline.vae.to("cpu")
    torch.cuda.empty_cache()

    guidance_scale = config["training"]["cfg"]["guidance_scale"]
    num_steps = 50
    latent_shape = None  # determined from first successful run

    from PIL import Image

    rng = torch.Generator(device=device).manual_seed(args.seed)

    for i, s in enumerate(missing):
        print(f"  [{i+1}/{len(missing)}] {s.object_id} ...")

        try:
            # Step 1: Encode image condition
            image = Image.open(s.input_image).convert("RGBA")
            cond_input = pipeline.prepare_image(image)
            img_tensor = cond_input["image"].to(device)
            mask_tensor = cond_input["mask"].to(device)

            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                contexts = cond_model(image=img_tensor, mask=mask_tensor)

            image_cond = contexts["main"]  # [1, N_tokens, D] on GPU

            # Step 2: Run teacher 50-step ODE to generate latent
            # Determine latent shape from the DiT's expected input
            if latent_shape is None:
                latent_shape = _get_latent_shape(teacher)
                print(f"  Latent shape: {latent_shape}")

            noise = torch.randn(1, *latent_shape, device=device, generator=rng)

            t0 = time.time()
            latent = teacher_ode_solve(
                teacher, noise, contexts,
                num_steps=num_steps, guidance_scale=guidance_scale,
            )
            elapsed = time.time() - t0

            # Save (squeeze batch dim, move to CPU)
            latent_np = latent.float().cpu().numpy()[0]
            image_cond_np = image_cond.float().cpu().numpy()[0]

            npz_path = os.path.join(out_dir, f"{s.object_id}.npz")
            np.savez_compressed(
                npz_path,
                latent=latent_np,
                image_cond=image_cond_np,
                object_id=s.object_id,
            )
            print(f"    Saved: latent {latent_np.shape}, cond {image_cond_np.shape} ({elapsed:.1f}s)")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"    FAILED: {e}")

    print("Done.")


def _get_latent_shape(teacher):
    """Determine the latent shape the DiT expects.

    Hunyuan3D-2.1 DiT operates on [num_latents, latent_dim] tokens.
    """
    # The DiT config stores these as attributes
    if hasattr(teacher, "num_latents") and hasattr(teacher, "in_channels"):
        return (teacher.num_latents, teacher.in_channels)

    # Fallback: inspect the input projection layer
    if hasattr(teacher, "x_embedder"):
        proj = teacher.x_embedder
        if hasattr(proj, "in_features"):
            latent_dim = proj.in_features
        elif hasattr(proj, "weight"):
            latent_dim = proj.weight.shape[1]
        else:
            latent_dim = 64
        return (4096, latent_dim)

    # Last resort
    return (4096, 64)


if __name__ == "__main__":
    main()
