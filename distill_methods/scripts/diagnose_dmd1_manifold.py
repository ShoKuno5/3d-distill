#!/usr/bin/env python3
"""E1: DMD1 latent volume-logit diagnosis.

For each test sample, runs DMD1 with `output_type="latent"`, saves the latent,
then runs the VAE + volume_decoder to obtain the 3D SDF grid and computes
on-manifold-ness statistics. Classifies each sample as:

  Mode A (decoder refusal):  saturated, neg% > 99.5, |max - min| < 0.05
  Mode B (generator drift):  valid zero-crossing, neg% <= 99.5, |max - min| > 0.05
  unclear:                   anything else

Usage (CWD must be models/hunyuan3d21/hy3dshape for hy3dshape imports):

    cd /mnt/workspace/kuno/distillation/models/hunyuan3d21/hy3dshape
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=2 \
      /mnt/workspace/kuno/distillation/envs/hunyuan3d-venv/bin/python \
      ../../../distill_methods/scripts/diagnose_dmd1_manifold.py \
      --config ../../../distill_methods/config_cross_family.yaml \
      --model-name dmd1_1step \
      --out-dir /mnt/workspace/kuno/distillation/results/distill_methods/runs/20260421_manifold_diagnosis/e1_dmd1_volume_logit
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_DIR / "pipeline"))

from src.utils.inference_config import (
    get_model_config,
    get_inference_params,
    load_and_filter_samples,
    resolve_model_paths,
)


EPS_SATURATE_RANGE = 0.05
EPS_NEG_PCT = 99.5


def classify_mode(min_v: float, max_v: float, neg_pct: float, zero_crossing: bool) -> str:
    span = max_v - min_v
    if (not zero_crossing) or (span < EPS_SATURATE_RANGE and neg_pct > EPS_NEG_PCT):
        return "A"  # decoder refusal
    if zero_crossing and neg_pct <= EPS_NEG_PCT and span >= EPS_SATURATE_RANGE:
        return "B"  # valid SDF, wrong-or-right shape (CD tells us which)
    return "unclear"


def compute_volume_stats(vae, latent_t, octree_resolution: int = 384):
    """Run VAE + volume_decoder on one latent, return stats dict + grid shape.

    `latent_t`: shape (1, T, D) torch tensor, any dtype; will be cast to VAE dtype.
    """
    import torch
    vae_dtype = next(vae.parameters()).dtype
    latent_t = latent_t.to(dtype=vae_dtype)
    with torch.no_grad():
        decoded = vae(latent_t / vae.scale_factor)
        grid = vae.volume_decoder(
            decoded, vae.geo_decoder,
            bounds=1.01, num_chunks=8000,
            octree_resolution=octree_resolution,
            enable_pbar=False,
        )
    g = grid[0].float().cpu().numpy()
    min_v = float(g.min())
    max_v = float(g.max())
    neg_pct = 100.0 * float((g < 0).mean())
    std_v = float(g.std())
    zero_crossing = bool(min_v < 0.0 and max_v > 0.0)
    return {
        "min": min_v,
        "max": max_v,
        "neg_pct": neg_pct,
        "std": std_v,
        "zero_crossing": zero_crossing,
        "grid_shape": tuple(int(x) for x in g.shape),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--octree-resolution", type=int, default=384)
    ap.add_argument("--skip-existing", action="store_true", default=True,
                    help="Skip samples whose latent .npy already exists")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    latent_dir = args.out_dir / "latents"
    latent_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    resolve_model_paths(cfg)

    samples = load_and_filter_samples(
        cfg, max_samples_override=args.max_samples, manifest_key="test_manifest"
    )
    model_cfg = get_model_config(cfg, args.model_name)
    if model_cfg is None:
        print(f"ERROR: '{args.model_name}' not in config")
        sys.exit(1)

    inf_params = get_inference_params(cfg, args.model_name)
    lora_path = inf_params.get("lora_path")
    num_steps = inf_params.get("num_inference_steps", 50)
    guidance_scale = inf_params.get("guidance_scale", 1.0)
    octree_resolution = args.octree_resolution

    print(f"[{args.model_name}] num_steps={num_steps} guidance={guidance_scale} octree={octree_resolution}")
    print(f"[{args.model_name}] lora_path={lora_path}")
    print(f"[{args.model_name}] samples={len(samples)}")

    import torch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    from PIL import Image

    print("Loading Hunyuan3D-2.1 pipeline ...")
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        cfg["training"]["model"]["pretrained"],
        use_safetensors=False,
    )

    if lora_path and os.path.exists(os.path.join(lora_path, "adapter_config.json")):
        print(f"Loading LoRA from {lora_path} ...")
        from peft import PeftModel
        dit = pipeline.model
        pipeline.model = PeftModel.from_pretrained(dit, lora_path)
        pipeline.model = pipeline.model.merge_and_unload()
        print("LoRA merged.")
    elif lora_path and os.path.exists(os.path.join(lora_path, "model.pt")):
        print(f"Loading full state_dict from {lora_path}/model.pt ...")
        state_dict = torch.load(os.path.join(lora_path, "model.pt"), map_location="cpu")
        pipeline.model.load_state_dict(state_dict)
        print("Full weights loaded.")

    csv_path = args.out_dir / "diagnosis.csv"
    fieldnames = [
        "object_id", "category", "seed",
        "min", "max", "neg_pct", "std", "zero_crossing", "span",
        "mode_class", "inference_sec", "status", "error",
    ]
    # Resume: load existing CSV to avoid redoing rows
    existing_rows = {}
    if csv_path.exists():
        with open(csv_path, "r") as f:
            for r in csv.DictReader(f):
                existing_rows[r["object_id"]] = r

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in existing_rows.values():
            writer.writerow(r)
        f.flush()

        for idx, s in enumerate(samples):
            oid = s.object_id
            out_latent = latent_dir / f"{oid}.npy"
            if args.skip_existing and oid in existing_rows and out_latent.exists():
                continue

            try:
                image = Image.open(s.input_image).convert("RGBA")
                torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
                latent = pipeline(
                    image=image,
                    generator=torch.manual_seed(args.seed),
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    output_type="latent",
                )
                # latent shape: (1, T, D) torch tensor on GPU
                t_infer = time.time() - t0
                latent_cpu = latent.float().cpu().numpy()[0]  # (T, D)
                np.save(out_latent, latent_cpu)

                stats = compute_volume_stats(pipeline.vae, latent, octree_resolution)
                span = stats["max"] - stats["min"]
                mode = classify_mode(stats["min"], stats["max"], stats["neg_pct"], stats["zero_crossing"])

                row = {
                    "object_id": oid,
                    "category": s.category,
                    "seed": args.seed,
                    "min": stats["min"],
                    "max": stats["max"],
                    "neg_pct": stats["neg_pct"],
                    "std": stats["std"],
                    "zero_crossing": stats["zero_crossing"],
                    "span": span,
                    "mode_class": mode,
                    "inference_sec": round(t_infer, 2),
                    "status": "success",
                    "error": "",
                }
            except Exception as e:
                import traceback
                traceback.print_exc()
                row = {
                    "object_id": oid,
                    "category": s.category,
                    "seed": args.seed,
                    "min": "", "max": "", "neg_pct": "", "std": "",
                    "zero_crossing": "", "span": "",
                    "mode_class": "error",
                    "inference_sec": "",
                    "status": "failed",
                    "error": str(e)[:200],
                }
            writer.writerow(row)
            f.flush()
            existing_rows[oid] = row

            print(
                f"[{idx+1:3d}/{len(samples)}] {oid:30s} "
                f"cat={s.category:12s} "
                f"mode={row['mode_class']:>7s} "
                f"neg%={row.get('neg_pct', '?')} "
                f"span={row.get('span', '?')} "
                f"t={row.get('inference_sec', '?')}s"
            )

    # Summary
    from collections import Counter
    modes = Counter(r["mode_class"] for r in existing_rows.values())
    print(f"\n=== Mode breakdown ({args.model_name}) ===")
    for k, v in sorted(modes.items()):
        print(f"  {k}: {v}")

    print(f"\nLatents: {latent_dir}")
    print(f"CSV:     {csv_path}")


if __name__ == "__main__":
    main()
