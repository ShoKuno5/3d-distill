#!/usr/bin/env python3
"""SAM-3D-Objects inference wrapper for aligned evaluation.

Reuses existing inference outputs when available.

Usage:
    CONDA_PREFIX=envs/sam3d-mamba/envs/sam3d-objects \
    CUDA_VISIBLE_DEVICES=2 \
    envs/sam3d-mamba/envs/sam3d-objects/bin/python \
    pipeline/scripts/run_inference_sam3d_objects.py \
    --config experiments/toys4k_baseline/config.yaml

To check existing outputs (no model env needed):
    python pipeline/scripts/run_inference_sam3d_objects.py \
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

from src.data.toys4k import load_manifest, filter_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    max_samples = args.max_samples or cfg["dataset"].get("max_samples")
    samples = load_manifest(cfg["dataset"]["manifest"])
    samples = filter_samples(samples, max_samples=max_samples)

    model_cfg = next(m for m in cfg["models"] if m["name"] == "sam3d")
    pred_root = model_cfg["predictions_root"]
    mesh_fn = model_cfg["mesh_filename"]

    existing = 0
    missing = []
    for s in samples:
        mesh_path = os.path.join(pred_root, s.object_id, mesh_fn)
        if os.path.exists(mesh_path):
            existing += 1
        else:
            missing.append(s)

    print(f"SAM-3D: {existing}/{len(samples)} predictions exist")

    if args.check_only:
        if missing:
            print(f"Missing: {[s.object_id for s in missing]}")
        return

    if not missing:
        print("All predictions available. Nothing to do.")
        return

    print(f"Running inference for {len(missing)} missing samples...")

    try:
        sam3d_dir = str(PROJECT_DIR.parent / "models" / "sam3d")
        os.environ.setdefault("CONDA_PREFIX",
            str(PROJECT_DIR.parent / "envs" / "sam3d-mamba" / "envs" / "sam3d-objects"))
        os.environ["LIDRA_SKIP_INIT"] = "true"

        import torch
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        import numpy as np
        from PIL import Image
        import trimesh
        import json
        import time

        sys.path.insert(0, os.path.join(sam3d_dir, "notebook"))
        sys.path.insert(0, sam3d_dir)
        from inference import Inference
    except ImportError as e:
        print(f"Cannot import SAM-3D: {e}")
        print("Run with SAM-3D conda env.")
        sys.exit(1)

    config_path = os.path.join(sam3d_dir, "checkpoints/hf/pipeline.yaml")
    infer = Inference(config_path, compile=False)

    for s in missing:
        out_dir = os.path.join(pred_root, s.object_id)
        os.makedirs(out_dir, exist_ok=True)
        print(f"  Processing: {s.object_id}")

        try:
            pil_img = Image.open(s.input_image).convert("RGBA")
            img_arr = np.array(pil_img)

            if img_arr.shape[2] == 4:
                rgb = img_arr[..., :3]
                mask = img_arr[..., 3] > 128
            else:
                rgb = img_arr[..., :3]
                mask = ~np.all(rgb > 240, axis=-1)

            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()

            output = infer(rgb, mask, seed=args.seed)

            mesh_obj = output.get("mesh")
            if mesh_obj is not None:
                _mesh = mesh_obj[0] if isinstance(mesh_obj, list) else mesh_obj
                raw_mesh = trimesh.Trimesh(
                    vertices=_mesh.vertices.cpu().numpy(),
                    faces=_mesh.faces.cpu().numpy(),
                    process=False,
                )
                raw_mesh.export(os.path.join(out_dir, mesh_fn))

            runtime = time.time() - t0
            peak_mem = torch.cuda.max_memory_allocated() / 1e9

            meta = {
                "model": "sam3d", "object_id": s.object_id,
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
