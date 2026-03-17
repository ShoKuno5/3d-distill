#!/usr/bin/env python3
"""TRELLIS inference wrapper for aligned evaluation.

Reuses existing inference outputs when available.
For new runs, calls the existing TRELLIS inference pipeline.

Usage (from repo root):
    cd models/trellis && PYTHONPATH=. \
    envs/miniconda3/envs/trellis/bin/python \
    ../pipeline/scripts/run_inference_trellis.py \
    --config ../experiments/toys4k_baseline/config.yaml

To use existing outputs (default — no TRELLIS env needed):
    python pipeline/scripts/run_inference_trellis.py \
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
    parser.add_argument("--check-only", action="store_true",
                        help="Only check which outputs exist, don't run inference")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    samples = load_and_filter_samples(cfg, max_samples_override=args.max_samples)

    # Find trellis model config
    model_cfg = get_model_config(cfg, "trellis")
    if model_cfg is None:
        print("ERROR: 'trellis' not found in config models list")
        sys.exit(1)
    pred_root = model_cfg["predictions_root"]
    mesh_fn = model_cfg["mesh_filename"]

    # Load inference params from config (falls back to defaults)
    inf_params = get_inference_params(cfg, "trellis")
    print(f"TRELLIS inference params: {inf_params}")

    existing = 0
    missing = []
    for s in samples:
        mesh_path = os.path.join(pred_root, s.object_id, mesh_fn)
        if os.path.exists(mesh_path):
            existing += 1
        else:
            missing.append(s)

    print(f"TRELLIS: {existing}/{len(samples)} predictions exist")

    if args.check_only:
        if missing:
            print(f"Missing: {[s.object_id for s in missing]}")
        return

    if not missing:
        print("All predictions available. Nothing to do.")
        return

    # Run inference for missing samples
    print(f"Running inference for {len(missing)} missing samples...")

    try:
        import torch
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['SPCONV_ALGO'] = 'native'
        from trellis.pipelines import TrellisImageTo3DPipeline
        from PIL import Image
        import trimesh
        import json
        import time
    except ImportError as e:
        print(f"Cannot import TRELLIS: {e}")
        print("Run from TRELLIS directory with proper PYTHONPATH.")
        sys.exit(1)

    pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipeline.cuda()

    CONFIG = inf_params

    for s in missing:
        out_dir = os.path.join(pred_root, s.object_id)
        os.makedirs(out_dir, exist_ok=True)
        print(f"  Processing: {s.object_id}")

        try:
            image = Image.open(s.input_image).convert("RGBA")
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()

            outputs = pipeline.run(
                image, seed=args.seed,
                sparse_structure_sampler_params={
                    "steps": CONFIG["ss_steps"],
                    "cfg_strength": CONFIG["ss_cfg"],
                },
                slat_sampler_params={
                    "steps": CONFIG["slat_steps"],
                    "cfg_strength": CONFIG["slat_cfg"],
                },
            )

            mesh_result = outputs["mesh"][0]
            raw_verts = mesh_result.vertices.cpu().numpy()
            raw_faces = mesh_result.faces.cpu().numpy()
            raw_mesh = trimesh.Trimesh(vertices=raw_verts, faces=raw_faces, process=False)
            raw_mesh.export(os.path.join(out_dir, mesh_fn))

            runtime = time.time() - t0
            peak_mem = torch.cuda.max_memory_allocated() / 1e9

            meta = {
                "model": "trellis", "inference_params": CONFIG, "object_id": s.object_id,
                "category": s.category, "input_image": s.input_image,
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
