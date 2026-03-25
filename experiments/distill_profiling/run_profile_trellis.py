#!/usr/bin/env python3
"""Block-level inference time profiling for TRELLIS v1 (teacher).

Expands TrellisImageTo3DPipeline.run() inline with block-level timing.
Saves profile.json + mesh_raw.obj per sample.

Usage (from models/trellis):
    PYTHONPATH=. SPCONV_ALGO=native CUDA_VISIBLE_DEVICES=0 \
    ../../envs/miniconda3/envs/trellis/bin/python \
    ../../experiments/distill_profiling/run_profile_trellis.py \
    --config ../../experiments/distill_profiling/config.yaml
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_DIR / "pipeline"))

from src.utils.inference_config import (
    get_inference_params,
    get_model_config,
    load_and_filter_samples,
)


def sync_and_time():
    import torch
    torch.cuda.synchronize()
    return time.perf_counter()


def profile_one_sample(pipeline, image, seed, ss_params, slat_params):
    """Run trellis v1 pipeline with block-level timing."""
    import torch

    timings = {}

    t0 = sync_and_time()
    image = pipeline.preprocess_image(image)
    timings["preprocess"] = sync_and_time() - t0

    torch.manual_seed(seed)

    t0 = sync_and_time()
    cond = pipeline.get_cond([image])
    timings["conditioning"] = sync_and_time() - t0

    t0 = sync_and_time()
    coords = pipeline.sample_sparse_structure(cond, 1, ss_params)
    timings["sparse_structure"] = sync_and_time() - t0

    t0 = sync_and_time()
    slat = pipeline.sample_slat(cond, coords, slat_params)
    timings["slat"] = sync_and_time() - t0

    t0 = sync_and_time()
    out = pipeline.decode_slat(slat, formats=["mesh"])
    timings["decode"] = sync_and_time() - t0

    return out, timings


def main():
    parser = argparse.ArgumentParser(description="TRELLIS v1 teacher profiling")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--sample-ids", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.sample_ids:
        cfg["dataset"]["sample_ids_file"] = args.sample_ids

    samples = load_and_filter_samples(cfg, max_samples_override=args.max_samples)
    model_cfg = get_model_config(cfg, "trellis")
    if model_cfg is None:
        print("ERROR: 'trellis' not found in config")
        sys.exit(1)

    pred_root = model_cfg["predictions_root"]
    mesh_fn = model_cfg["mesh_filename"]
    inf_params = get_inference_params(cfg, "trellis")

    ss_params = {
        "steps": inf_params["ss_steps"],
        "cfg_strength": inf_params["ss_cfg"],
    }
    slat_params = {
        "steps": inf_params["slat_steps"],
        "cfg_strength": inf_params["slat_cfg"],
    }

    print(f"TRELLIS v1 teacher profiling | {len(samples)} samples | "
          f"ss_steps={inf_params['ss_steps']} slat_steps={inf_params['slat_steps']}")

    import torch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["SPCONV_ALGO"] = "native"
    from trellis.pipelines import TrellisImageTo3DPipeline
    from PIL import Image
    import trimesh

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    print("Loading TRELLIS v1 pipeline...")
    pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipeline.cuda()

    # Warmup
    if samples:
        warmup_s = samples[0]
        print(f"  [WARMUP] {warmup_s.object_id}")
        warmup_img = Image.open(warmup_s.input_image).convert("RGBA")
        with torch.no_grad():
            profile_one_sample(pipeline, warmup_img, args.seed, ss_params, slat_params)
        torch.cuda.empty_cache()
        print("  [WARMUP] done")

    for i, s in enumerate(samples):
        out_dir = os.path.join(pred_root, s.object_id)
        os.makedirs(out_dir, exist_ok=True)

        print(f"  [{i + 1}/{len(samples)}] {s.object_id}")

        try:
            image = Image.open(s.input_image).convert("RGBA")
            torch.cuda.reset_peak_memory_stats()

            t_total_start = sync_and_time()
            with torch.no_grad():
                out, block_timings = profile_one_sample(
                    pipeline, image, args.seed, ss_params, slat_params)
            t_total = sync_and_time() - t_total_start

            mesh_result = out["mesh"][0]
            peak_mem = torch.cuda.max_memory_allocated() / 1e9

            # Save mesh
            raw_verts = mesh_result.vertices.cpu().numpy()
            raw_faces = mesh_result.faces.cpu().numpy()
            trimesh.Trimesh(vertices=raw_verts, faces=raw_faces, process=False).export(
                os.path.join(out_dir, mesh_fn))

            profile = {
                "object_id": s.object_id,
                "category": s.category,
                "model": "trellis",
                "seed": args.seed,
                "blocks": {k: {"wall_sec": round(v, 4)} for k, v in block_timings.items()},
                "blocks_sum_sec": round(sum(block_timings.values()), 4),
                "total_sec": round(t_total, 4),
                "peak_gpu_gb": round(peak_mem, 2),
                "raw_vertices": int(mesh_result.vertices.shape[0]),
                "raw_faces": int(mesh_result.faces.shape[0]),
            }

            with open(os.path.join(out_dir, "profile.json"), "w") as f:
                json.dump(profile, f, indent=2)

            block_str = " | ".join(f"{k}={v:.2f}s" for k, v in block_timings.items())
            print(f"    total={t_total:.2f}s | {block_str}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"    FAILED: {e}")
            profile = {
                "object_id": s.object_id,
                "model": "trellis",
                "status": "failed",
                "error": str(e),
            }
            with open(os.path.join(out_dir, "profile.json"), "w") as f:
                json.dump(profile, f, indent=2)


if __name__ == "__main__":
    main()
