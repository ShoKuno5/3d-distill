#!/usr/bin/env python3
"""Hunyuan3D-2 inference wrapper for aligned evaluation.

Reuses existing inference outputs when available.

Usage:
    cd models/hunyuan3d && CUDA_VISIBLE_DEVICES=1 \
    LD_PRELOAD=/opt/unreal-engine/usr/lib/x86_64-linux-gnu/libOpenGL.so.0 \
    venv/bin/python ../pipeline/scripts/run_inference_hunyuan3d2.py \
    --config ../experiments/toys4k_baseline/config.yaml

To check existing outputs (no model env needed):
    python pipeline/scripts/run_inference_hunyuan3d2.py \
    --config experiments/toys4k_baseline/config.yaml --check-only
"""

import argparse
import os
import sys
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.utils.inference_config import get_model_config, get_inference_params, load_and_filter_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    samples = load_and_filter_samples(cfg, max_samples_override=args.max_samples)

    model_cfg = get_model_config(cfg, "hunyuan3d")
    if model_cfg is None:
        print("ERROR: 'hunyuan3d' not found in config models list")
        sys.exit(1)
    pred_root = model_cfg["predictions_root"]
    mesh_fn = model_cfg["mesh_filename"]

    # Load inference params from config (falls back to defaults)
    inf_params = get_inference_params(cfg, "hunyuan3d")
    print(f"Hunyuan3D inference params: {inf_params}")

    existing = 0
    missing = []
    for s in samples:
        mesh_path = os.path.join(pred_root, s.object_id, mesh_fn)
        if os.path.exists(mesh_path):
            existing += 1
        else:
            missing.append(s)

    print(f"Hunyuan3D: {existing}/{len(samples)} predictions exist")

    if args.check_only:
        if missing:
            print(f"Missing: {[s.object_id for s in missing]}")
        return

    if not missing:
        print("All predictions available. Nothing to do.")
        return

    print(f"Running inference for {len(missing)} missing samples...")

    try:
        import torch
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        from PIL import Image
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        import json
        import time
    except ImportError as e:
        print(f"Cannot import Hunyuan3D: {e}")
        print("Run from models/hunyuan3d directory with proper venv.")
        sys.exit(1)

    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2')

    for s in missing:
        out_dir = os.path.join(pred_root, s.object_id)
        os.makedirs(out_dir, exist_ok=True)
        print(f"  Processing: {s.object_id}")

        try:
            image = Image.open(s.input_image).convert("RGBA")
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()

            mesh = pipeline(
                image=image,
                generator=torch.manual_seed(args.seed),
                num_inference_steps=inf_params["num_inference_steps"],
                guidance_scale=inf_params["guidance_scale"],
                octree_resolution=inf_params["octree_resolution"],
            )[0]
            mesh.export(os.path.join(out_dir, mesh_fn))

            runtime = time.time() - t0
            peak_mem = torch.cuda.max_memory_allocated() / 1e9

            meta = {
                "model": "hunyuan3d", "object_id": s.object_id,
                "category": s.category, "input_image": s.input_image,
                "inference_params": inf_params,
                "runtime_sec": round(runtime, 2), "peak_gpu_gb": round(peak_mem, 2),
                "seed": args.seed, "status": "success",
            }
            with open(os.path.join(out_dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
            print(f"    Done: {runtime:.1f}s")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"    FAILED: {e}")


if __name__ == "__main__":
    main()
