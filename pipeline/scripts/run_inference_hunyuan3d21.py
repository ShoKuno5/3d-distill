#!/usr/bin/env python3
"""Hunyuan3D-2.1 inference wrapper for aligned evaluation.

Hunyuan3D-2.1 is the updated version with PBR texture support and
improved shape generation. For geometry-only evaluation, we use
the shape pipeline only (no paint stage).

Key differences from Hunyuan3D-2.0:
  - Module: hy3dshape.pipelines (not hy3dgen.shapegen)
  - Model: tencent/Hunyuan3D-2.1 (not tencent/Hunyuan3D-2)
  - Default output: GLB (we save OBJ via trimesh for consistency)
  - Has built-in rembg

Usage (from Hunyuan3D-2.1 repo root):
    cd models/hunyuan3d21 && \
    PYTHONPATH=. \
    LD_PRELOAD=/opt/unreal-engine/usr/lib/x86_64-linux-gnu/libOpenGL.so.0 \
    CUDA_VISIBLE_DEVICES=1 \
    envs/hunyuan3d-venv/bin/python \
    ../pipeline/scripts/run_inference_hunyuan3d21.py \
    --config ../experiments/toys4k_baseline/config.yaml

To check existing outputs (no model env needed):
    python pipeline/scripts/run_inference_hunyuan3d21.py \
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
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU device index")
    parser.add_argument("--sample-ids", type=str, default=None,
                        help="Path to sample IDs file (overrides config sample_ids_file)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.sample_ids:
        cfg["dataset"]["sample_ids_file"] = args.sample_ids

    samples = load_and_filter_samples(cfg, max_samples_override=args.max_samples)

    # Find hunyuan3d21 model config
    model_cfg = get_model_config(cfg, "hunyuan3d21")
    if model_cfg is None:
        print("ERROR: 'hunyuan3d21' not found in config models list")
        sys.exit(1)

    pred_root = model_cfg["predictions_root"]
    mesh_fn = model_cfg["mesh_filename"]

    # Load inference params from config (falls back to defaults)
    inf_params = get_inference_params(cfg, "hunyuan3d21")
    print(f"Hunyuan3D-2.1 inference params: {inf_params}")

    existing = 0
    missing = []
    for s in samples:
        mesh_path = os.path.join(pred_root, s.object_id, mesh_fn)
        if os.path.exists(mesh_path):
            existing += 1
        else:
            missing.append(s)

    print(f"Hunyuan3D-2.1: {existing}/{len(samples)} predictions exist")

    if args.check_only:
        if missing:
            print(f"Missing: {[s.object_id for s in missing]}")
        return

    if not missing:
        print("All predictions available. Nothing to do.")
        return

    print(f"Running inference for {len(missing)} missing samples...")

    try:
        # Hunyuan3D-2.1: demo.py does sys.path.insert(0, './hy3dshape')
        # then imports from hy3dshape.pipelines (package is at hy3dshape/hy3dshape/)
        hy3dshape_parent = os.path.join(os.getcwd(), "hy3dshape")
        if hy3dshape_parent not in sys.path:
            sys.path.insert(0, hy3dshape_parent)
        if os.getcwd() not in sys.path:
            sys.path.insert(0, os.getcwd())

        import torch
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        from PIL import Image
        import trimesh
        import json
        import time

        # Try torchvision fix if available
        try:
            from torchvision_fix import apply_fix
            apply_fix()
        except (ImportError, Exception):
            pass

        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    except ImportError as e:
        print(f"Cannot import Hunyuan3D-2.1: {e}")
        print("Run from models/hunyuan3d21 directory with proper environment.")
        print("  cd models/hunyuan3d21 && PYTHONPATH=.")
        sys.exit(1)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2.1')

    for s in missing:
        out_dir = os.path.join(pred_root, s.object_id)
        os.makedirs(out_dir, exist_ok=True)
        print(f"  Processing: {s.object_id}")

        try:
            image = Image.open(s.input_image).convert("RGBA")
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()

            # Same API as 2.0: returns list of trimesh objects
            mesh = pipeline(
                image=image,
                generator=torch.manual_seed(args.seed),
                num_inference_steps=inf_params["num_inference_steps"],
                guidance_scale=inf_params["guidance_scale"],
                octree_resolution=inf_params["octree_resolution"],
            )[0]

            # Hunyuan3D-2.1 may return trimesh.Trimesh or similar
            # Export to OBJ for consistent evaluation
            if hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
                if hasattr(mesh.vertices, 'cpu'):
                    # torch tensor
                    verts = mesh.vertices.cpu().numpy()
                    faces = mesh.faces.cpu().numpy()
                    raw_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
                elif isinstance(mesh, trimesh.Trimesh):
                    raw_mesh = mesh
                else:
                    # numpy array already
                    raw_mesh = trimesh.Trimesh(
                        vertices=mesh.vertices, faces=mesh.faces, process=False
                    )
            elif isinstance(mesh, trimesh.Scene):
                raw_mesh = mesh.dump(concatenate=True)
            else:
                raw_mesh = mesh

            raw_mesh.export(os.path.join(out_dir, mesh_fn))

            runtime = time.time() - t0
            peak_mem = torch.cuda.max_memory_allocated() / 1e9

            meta = {
                "model": "hunyuan3d21",
                "model_id": "tencent/Hunyuan3D-2.1",
                "object_id": s.object_id,
                "category": s.category,
                "input_image": s.input_image,
                "inference_params": inf_params,
                "runtime_sec": round(runtime, 2),
                "peak_gpu_gb": round(peak_mem, 2),
                "seed": args.seed,
                "status": "success",
                "raw_vertices": int(len(raw_mesh.vertices)),
                "raw_faces": int(len(raw_mesh.faces)),
            }
            with open(os.path.join(out_dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
            print(f"    Done: {runtime:.1f}s, {len(raw_mesh.vertices)} verts, {len(raw_mesh.faces)} faces")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"    FAILED: {e}")
            meta = {
                "model": "hunyuan3d21",
                "object_id": s.object_id,
                "status": "failed",
                "error": str(e),
            }
            with open(os.path.join(out_dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
