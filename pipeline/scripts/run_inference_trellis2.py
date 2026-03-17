#!/usr/bin/env python3
"""TRELLIS.2 inference wrapper for aligned evaluation.

TRELLIS.2 is a 4B-parameter model using O-Voxel representation.
Returns MeshWithVoxel with .vertices/.faces (torch tensors).

Usage (from repo root):
    cd models/trellis2 && \
    CUDA_HOME=/usr/local/cuda-12.4 \
    envs/miniconda3/envs/trellis2/bin/python \
    ../pipeline/scripts/run_inference_trellis2.py \
    --config ../experiments/toys4k_baseline/config.yaml

To check existing outputs (no trellis2 env needed):
    python pipeline/scripts/run_inference_trellis2.py \
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
    parser.add_argument("--pipeline-type", type=str, default=None,
                        help="TRELLIS.2 pipeline type (overrides config)")
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU device index")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    samples = load_and_filter_samples(cfg, max_samples_override=args.max_samples)

    # Find trellis2 model config
    model_cfg = get_model_config(cfg, "trellis2")
    if model_cfg is None:
        print("ERROR: 'trellis2' not found in config models list")
        sys.exit(1)

    pred_root = model_cfg["predictions_root"]
    mesh_fn = model_cfg["mesh_filename"]

    # Load inference params from config (falls back to defaults)
    inf_params = get_inference_params(cfg, "trellis2")
    # CLI --pipeline-type overrides config
    if args.pipeline_type is not None:
        inf_params["pipeline_type"] = args.pipeline_type
    pipeline_type = inf_params["pipeline_type"]
    print(f"TRELLIS.2 inference params: {inf_params}")

    existing = 0
    missing = []
    for s in samples:
        mesh_path = os.path.join(pred_root, s.object_id, mesh_fn)
        if os.path.exists(mesh_path):
            existing += 1
        else:
            missing.append(s)

    print(f"TRELLIS.2: {existing}/{len(samples)} predictions exist")

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
        from trellis2.pipelines import Trellis2ImageTo3DPipeline
        from PIL import Image
        import trimesh
        import json
        import time
    except ImportError as e:
        print(f"Cannot import TRELLIS.2: {e}")
        print("Run from models/trellis2 directory with proper environment.")
        print("  cd models/trellis2 && conda activate trellis2")
        sys.exit(1)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    # The pretrained config references gated briaai/RMBG-2.0 for rembg.
    # Override with the open ZhengPeng7/BiRefNet which has the same interface.
    import trellis2.pipelines.rembg as _rembg_mod
    _OrigBiRefNet = _rembg_mod.BiRefNet
    _rembg_mod.BiRefNet = lambda **kwargs: _OrigBiRefNet(model_name="ZhengPeng7/BiRefNet")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    _rembg_mod.BiRefNet = _OrigBiRefNet  # restore
    pipeline.cuda()

    for s in missing:
        out_dir = os.path.join(pred_root, s.object_id)
        os.makedirs(out_dir, exist_ok=True)
        print(f"  Processing: {s.object_id}")

        try:
            image = Image.open(s.input_image).convert("RGBA")
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()

            # TRELLIS.2 run returns List[MeshWithVoxel]
            meshes = pipeline.run(
                image,
                seed=args.seed,
                pipeline_type=pipeline_type,
            )
            mesh_result = meshes[0]

            # Extract raw vertices and faces (torch tensors)
            raw_verts = mesh_result.vertices.cpu().numpy()
            raw_faces = mesh_result.faces.cpu().numpy()

            # Save as OBJ for consistent evaluation
            raw_mesh = trimesh.Trimesh(
                vertices=raw_verts, faces=raw_faces, process=False
            )
            raw_mesh.export(os.path.join(out_dir, mesh_fn))

            runtime = time.time() - t0
            peak_mem = torch.cuda.max_memory_allocated() / 1e9

            meta = {
                "model": "trellis2",
                "model_id": "microsoft/TRELLIS.2-4B",
                "inference_params": inf_params,
                "pipeline_type": pipeline_type,
                "object_id": s.object_id,
                "category": s.category,
                "input_image": s.input_image,
                "runtime_sec": round(runtime, 2),
                "peak_gpu_gb": round(peak_mem, 2),
                "seed": args.seed,
                "status": "success",
                "raw_vertices": int(raw_verts.shape[0]),
                "raw_faces": int(raw_faces.shape[0]),
            }
            with open(os.path.join(out_dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
            print(f"    Done: {runtime:.1f}s, {raw_verts.shape[0]} verts, {raw_faces.shape[0]} faces")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"    FAILED: {e}")
            meta = {
                "model": "trellis2",
                "object_id": s.object_id,
                "status": "failed",
                "error": str(e),
            }
            with open(os.path.join(out_dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
