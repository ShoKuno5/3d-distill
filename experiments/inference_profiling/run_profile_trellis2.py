#!/usr/bin/env python3
"""Block-level inference time profiling for TRELLIS.2.

Expands pipeline.run() inline to insert torch.cuda.synchronize() + time.perf_counter()
around each pipeline block. Saves per-sample profile.json with block-level wall times.

Usage (from models/trellis2):
    PYTHONPATH=. CUDA_HOME=../../envs/cuda-12.8 \
    ../../envs/miniconda3/envs/trellis2/bin/python \
    ../../experiments/inference_profiling/run_profile_trellis2.py \
    --config ../../experiments/inference_profiling/config.yaml
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
    """Synchronize CUDA and return perf_counter timestamp."""
    import torch
    torch.cuda.synchronize()
    return time.perf_counter()


def profile_one_sample(pipeline, image, seed, pipeline_type):
    """Run trellis2 pipeline with block-level timing.

    Replicates the logic of Trellis2ImageTo3DPipeline.run() with timers
    inserted between each block.
    """
    import torch

    timings = {}

    # --- Preprocess ---
    t0 = sync_and_time()
    image = pipeline.preprocess_image(image)
    timings["preprocess"] = sync_and_time() - t0

    torch.manual_seed(seed)

    # --- Conditioning (512) ---
    t0 = sync_and_time()
    cond_512 = pipeline.get_cond([image], 512)
    timings["cond_512"] = sync_and_time() - t0

    # --- Conditioning (1024) ---
    if pipeline_type != "512":
        t0 = sync_and_time()
        cond_1024 = pipeline.get_cond([image], 1024)
        timings["cond_1024"] = sync_and_time() - t0
    else:
        cond_1024 = None
        timings["cond_1024"] = 0.0

    # --- Sparse structure sampling ---
    ss_res = {"512": 32, "1024": 64, "1024_cascade": 32, "1536_cascade": 32}[pipeline_type]
    t0 = sync_and_time()
    coords = pipeline.sample_sparse_structure(cond_512, ss_res, 1, {})
    timings["sparse_structure"] = sync_and_time() - t0

    # --- Shape SLat sampling ---
    t0 = sync_and_time()
    if pipeline_type == "512":
        shape_slat = pipeline.sample_shape_slat(
            cond_512, pipeline.models["shape_slat_flow_model_512"], coords, {}
        )
        res = 512
    elif pipeline_type == "1024":
        shape_slat = pipeline.sample_shape_slat(
            cond_1024, pipeline.models["shape_slat_flow_model_1024"], coords, {}
        )
        res = 1024
    elif pipeline_type == "1024_cascade":
        shape_slat, res = pipeline.sample_shape_slat_cascade(
            cond_512, cond_1024,
            pipeline.models["shape_slat_flow_model_512"],
            pipeline.models["shape_slat_flow_model_1024"],
            512, 1024, coords, {},
        )
    elif pipeline_type == "1536_cascade":
        shape_slat, res = pipeline.sample_shape_slat_cascade(
            cond_512, cond_1024,
            pipeline.models["shape_slat_flow_model_512"],
            pipeline.models["shape_slat_flow_model_1024"],
            512, 1536, coords, {},
        )
    timings["shape_slat"] = sync_and_time() - t0

    # --- Texture SLat sampling ---
    cond_tex = cond_1024 if pipeline_type != "512" else cond_512
    tex_model_key = (
        "tex_slat_flow_model_512" if pipeline_type == "512"
        else "tex_slat_flow_model_1024"
    )
    t0 = sync_and_time()
    tex_slat = pipeline.sample_tex_slat(
        cond_tex, pipeline.models[tex_model_key], shape_slat, {}
    )
    timings["tex_slat"] = sync_and_time() - t0

    # --- Decode latent (shape VAE + texture VAE + mesh) ---
    torch.cuda.empty_cache()
    t0 = sync_and_time()
    out_mesh = pipeline.decode_latent(shape_slat, tex_slat, res)
    timings["decode"] = sync_and_time() - t0

    return out_mesh, timings


def main():
    parser = argparse.ArgumentParser(description="Trellis2 block-level profiling")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pipeline-type", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--sample-ids", type=str, default=None)
    parser.add_argument("--save-mesh", action="store_true",
                        help="Save mesh OBJ (default: profile only)")
    parser.add_argument("--warmup-image", type=str, default=None,
                        help="Path to image for warmup run (not recorded)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.sample_ids:
        cfg["dataset"]["sample_ids_file"] = args.sample_ids

    samples = load_and_filter_samples(cfg, max_samples_override=args.max_samples)
    model_cfg = get_model_config(cfg, "trellis2")
    if model_cfg is None:
        print("ERROR: 'trellis2' not found in config")
        sys.exit(1)

    pred_root = model_cfg["predictions_root"]
    mesh_fn = model_cfg["mesh_filename"]

    inf_params = get_inference_params(cfg, "trellis2")
    if args.pipeline_type is not None:
        inf_params["pipeline_type"] = args.pipeline_type
    pipeline_type = inf_params["pipeline_type"]
    print(f"TRELLIS.2 profiling | pipeline_type={pipeline_type} | {len(samples)} samples")

    # Import model dependencies
    import torch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from PIL import Image

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    # Load pipeline
    import trellis2.pipelines.rembg as _rembg_mod
    _OrigBiRefNet = _rembg_mod.BiRefNet
    _rembg_mod.BiRefNet = lambda **kwargs: _OrigBiRefNet(model_name="ZhengPeng7/BiRefNet")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    _rembg_mod.BiRefNet = _OrigBiRefNet
    pipeline.cuda()

    # Warmup run: warms CUDA caches, not recorded.
    # Use first manifest sample as warmup to match real workload (RGBA with alpha).
    # The warmup result is discarded.
    if samples:
        warmup_s = samples[0]
        print(f"  [WARMUP] {warmup_s.object_id}")
        warmup_img = Image.open(warmup_s.input_image).convert("RGBA")
        with torch.no_grad():
            profile_one_sample(pipeline, warmup_img, args.seed, pipeline_type)
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
                mesh_result_list, block_timings = profile_one_sample(
                    pipeline, image, args.seed, pipeline_type
                )
            t_total = sync_and_time() - t_total_start

            mesh_result = mesh_result_list[0]
            peak_mem = torch.cuda.max_memory_allocated() / 1e9

            profile = {
                "object_id": s.object_id,
                "category": s.category,
                "model": "trellis2",
                "pipeline_type": pipeline_type,
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

            if args.save_mesh:
                import trimesh
                raw_verts = mesh_result.vertices.cpu().numpy()
                raw_faces = mesh_result.faces.cpu().numpy()
                trimesh.Trimesh(vertices=raw_verts, faces=raw_faces, process=False).export(
                    os.path.join(out_dir, mesh_fn)
                )

            block_str = " | ".join(f"{k}={v:.2f}s" for k, v in block_timings.items())
            print(f"    total={t_total:.2f}s | {block_str}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"    FAILED: {e}")
            profile = {
                "object_id": s.object_id,
                "model": "trellis2",
                "status": "failed",
                "error": str(e),
            }
            with open(os.path.join(out_dir, "profile.json"), "w") as f:
                json.dump(profile, f, indent=2)


if __name__ == "__main__":
    main()
