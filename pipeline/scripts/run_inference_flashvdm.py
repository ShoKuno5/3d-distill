#!/usr/bin/env python3
"""FlashVDM inference wrapper for aligned evaluation.

Loads Hunyuan3D-2.0 turbo DiT pipeline with FlashVDM decoder acceleration.

Usage:
    cd models/hunyuan3d && \
    LD_PRELOAD=/opt/unreal-engine/usr/lib/x86_64-linux-gnu/libOpenGL.so.0 \
    venv/bin/python ../../pipeline/scripts/run_inference_flashvdm.py \
    --config ../../experiments/distill_comparison/config.yaml
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

    model_cfg = get_model_config(cfg, "flashvdm")
    if model_cfg is None:
        print("ERROR: 'flashvdm' not found in config models list")
        sys.exit(1)
    pred_root = model_cfg["predictions_root"]
    mesh_fn = model_cfg["mesh_filename"]

    inf_params = get_inference_params(cfg, "flashvdm")
    print(f"FlashVDM inference params: {inf_params}")

    existing = 0
    missing = []
    for s in samples:
        mesh_path = os.path.join(pred_root, s.object_id, mesh_fn)
        if os.path.exists(mesh_path):
            existing += 1
        else:
            missing.append(s)

    print(f"FlashVDM: {existing}/{len(samples)} predictions exist")

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

    # Load turbo DiT pipeline and enable FlashVDM
    print("Loading Hunyuan3D-2.0 turbo DiT pipeline...")
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2",
        subfolder="hunyuan3d-dit-v2-0-turbo",
        use_safetensors=True,
    )
    pipeline.enable_flashvdm(mc_algo="mc")
    print("FlashVDM enabled.")

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
                octree_resolution=inf_params["octree_resolution"],
                num_chunks=inf_params.get("num_chunks", 200000),
                output_type="trimesh",
            )[0]
            mesh.export(os.path.join(out_dir, mesh_fn))

            runtime = time.time() - t0
            peak_mem = torch.cuda.max_memory_allocated() / 1e9

            meta = {
                "model": "flashvdm", "object_id": s.object_id,
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
