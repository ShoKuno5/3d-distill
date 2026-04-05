#!/usr/bin/env python3
"""Generate (noise, x_teacher) regression pairs for DMD1 distillation.

Runs the teacher's 50-step ODE solver on random noise to create
pre-computed regression targets.

Usage:
    cd /mnt/workspace/kuno/distillation
    CUDA_VISIBLE_DEVICES=0 envs/hunyuan3d-venv/bin/python \
        distill_methods/scripts/generate_dmd1_pairs.py \
        --config distill_methods/config.yaml \
        --num-pairs 20000
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_DIR / "pipeline"))
sys.path.insert(0, str(PROJECT_DIR / "models" / "hunyuan3d21" / "hy3dshape"))


def teacher_ode_solve(model, noise, contexts, num_steps=50, guidance_scale=7.5):
    """Run the teacher's Euler ODE solver from t=0 (noise) to t=1 (data).

    Args:
        model: Teacher DiT model (frozen).
        noise: Initial noise [B, N, C].
        contexts: Conditioning dict.
        num_steps: Number of Euler steps.
        guidance_scale: CFG scale.

    Returns:
        x_1: Final denoised output [B, N, C].
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
    parser = argparse.ArgumentParser(description="Generate DMD1 regression pairs")
    parser.add_argument("--config", required=True)
    parser.add_argument("--num-pairs", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--shard", type=int, default=0, help="Shard index for multi-GPU")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    dmd1_cfg = config["training"]["methods"]["dmd1"]
    pairs_dir = dmd1_cfg["pairs_dir"]
    os.makedirs(pairs_dir, exist_ok=True)

    # Count existing
    existing = len([f for f in os.listdir(pairs_dir) if f.endswith(".npz")])
    remaining = args.num_pairs - existing
    if remaining <= 0:
        print(f"Already have {existing} pairs (target: {args.num_pairs}). Done.")
        return

    # Shard the remaining work across GPUs
    if args.num_shards > 1:
        shard_size = remaining // args.num_shards
        shard_start = existing + args.shard * shard_size
        shard_end = existing + (args.shard + 1) * shard_size if args.shard < args.num_shards - 1 else args.num_pairs
        print(f"Shard {args.shard}/{args.num_shards}: generating pairs {shard_start}-{shard_end} ...")
    else:
        shard_start = existing
        shard_end = args.num_pairs

    print(f"Generating {shard_end - shard_start} pairs (have {existing}, target {args.num_pairs}) ...")

    # Load teacher model
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    pretrained = config["training"]["model"]["pretrained"]
    print(f"Loading teacher from {pretrained} ...")
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        pretrained, use_safetensors=False,
    )
    teacher = pipeline.model
    device = torch.device("cuda")
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # We need a representative conditioning for generating pairs.
    # Load pre-encoded training data to get image_cond samples.
    training_data_dir = config["training"]["training_data_dir"]
    cond_files = sorted([
        f for f in os.listdir(training_data_dir) if f.endswith(".npz")
    ])
    if not cond_files:
        print(f"ERROR: No training data in {training_data_dir}. Run prepare_training_data.py first.")
        sys.exit(1)

    # Load all conditions into memory for random sampling
    all_conds = []
    latent_shape = None
    for f in cond_files:
        data = np.load(os.path.join(training_data_dir, f))
        all_conds.append(data["image_cond"])
        if latent_shape is None:
            latent_shape = data["latent"].shape
    print(f"Loaded {len(all_conds)} conditions, latent shape: {latent_shape}")

    guidance_scale = config["training"]["cfg"]["guidance_scale"]
    num_steps = 50
    pair_idx = shard_start

    while pair_idx < shard_end:
        B = min(args.batch_size, shard_end - pair_idx)

        # Random conditions
        cond_indices = np.random.randint(0, len(all_conds), B)
        image_cond = np.stack([all_conds[i] for i in cond_indices])
        image_cond_t = torch.from_numpy(image_cond).to(device)
        contexts = {"main": image_cond_t}

        # Random noise matching latent shape (latent_shape is per-sample, e.g. (4096, 64))
        noise = torch.randn(B, *latent_shape, device=device, dtype=torch.float32)

        # Teacher 50-step ODE
        x_teacher = teacher_ode_solve(
            teacher, noise, contexts,
            num_steps=num_steps, guidance_scale=guidance_scale,
        )

        # Save pairs
        noise_np = noise.cpu().numpy()
        x_teacher_np = x_teacher.cpu().numpy()
        for j in range(B):
            np.savez_compressed(
                os.path.join(pairs_dir, f"pair_{pair_idx:06d}.npz"),
                noise=noise_np[j],
                x_teacher=x_teacher_np[j],
                image_cond=image_cond[j],
            )
            pair_idx += 1

        if pair_idx % 100 == 0:
            print(f"  Shard {args.shard}: generated {pair_idx - shard_start}/{shard_end - shard_start} pairs (global idx {pair_idx})")

    print(f"Shard {args.shard} done. Generated pairs {shard_start}-{pair_idx} in {pairs_dir}")


if __name__ == "__main__":
    main()
