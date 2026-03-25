#!/usr/bin/env python3
"""Block-level inference time profiling for MDT-dist (TRELLIS v1 distilled).

Loads TRELLIS v1 pipeline, swaps weights with MDT-dist checkpoints,
replaces samplers with 2-step FlowEulerGuidanceIntervalSampler,
then runs inference with block-level timing.

Usage (from models/trellis):
    PYTHONPATH=. SPCONV_ALGO=native CUDA_VISIBLE_DEVICES=0 \
    ../../envs/miniconda3/envs/trellis/bin/python \
    ../../experiments/distill_profiling/run_profile_mdt_dist.py \
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
    """Synchronize CUDA and return perf_counter timestamp."""
    import torch
    torch.cuda.synchronize()
    return time.perf_counter()


def profile_one_sample(pipeline, image, seed):
    """Run mdt_dist pipeline with block-level timing.

    Replicates TrellisImageTo3DPipeline.run() with timers between each block.
    """
    import torch

    timings = {}

    # --- Preprocess ---
    t0 = sync_and_time()
    image = pipeline.preprocess_image(image)
    timings["preprocess"] = sync_and_time() - t0

    torch.manual_seed(seed)

    # --- Conditioning ---
    t0 = sync_and_time()
    cond = pipeline.get_cond([image])
    timings["conditioning"] = sync_and_time() - t0

    # --- Sparse structure sampling ---
    t0 = sync_and_time()
    coords = pipeline.sample_sparse_structure(cond, 1, pipeline.sparse_structure_sampler_params)
    timings["sparse_structure"] = sync_and_time() - t0

    # --- SLAT sampling ---
    t0 = sync_and_time()
    slat = pipeline.sample_slat(cond, coords, pipeline.slat_sampler_params)
    timings["slat"] = sync_and_time() - t0

    # --- Decode ---
    t0 = sync_and_time()
    out = pipeline.decode_slat(slat, formats=["mesh"])
    timings["decode"] = sync_and_time() - t0

    return out, timings


def main():
    parser = argparse.ArgumentParser(description="MDT-dist block-level profiling")
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
    model_cfg = get_model_config(cfg, "mdt_dist")
    if model_cfg is None:
        print("ERROR: 'mdt_dist' not found in config")
        sys.exit(1)

    pred_root = model_cfg["predictions_root"]
    mesh_fn = model_cfg["mesh_filename"]
    inf_params = get_inference_params(cfg, "mdt_dist")

    print(f"MDT-dist profiling | {len(samples)} samples | "
          f"ss_steps={inf_params['ss_steps']} slat_steps={inf_params['slat_steps']}")

    import torch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["SPCONV_ALGO"] = "native"
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.pipelines.samplers import FlowEulerGuidanceIntervalSampler
    from PIL import Image
    import trimesh

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    # Load TRELLIS v1 pipeline
    print("Loading TRELLIS v1 pipeline...")
    pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipeline.cuda()

    # Swap weights with MDT-dist checkpoints
    mdt_ckpts = inf_params.get("ckpts_dir",
        str(PROJECT_DIR / "models" / "mdt_dist" / "ckpts"))
    print(f"Swapping weights from {mdt_ckpts}...")

    ss_flow_state = torch.load(
        os.path.join(mdt_ckpts, "ss_flow_img_dit_L_16l8_fp16.pt"),
        map_location="cpu", weights_only=True)
    slat_flow_state = torch.load(
        os.path.join(mdt_ckpts, "slat_flow_img_dit_L_64l8p2_fp16.pt"),
        map_location="cpu", weights_only=True)

    pipeline.models["sparse_structure_flow_model"].load_state_dict(ss_flow_state)
    pipeline.models["slat_flow_model"].load_state_dict(slat_flow_state)
    pipeline.models["sparse_structure_flow_model"].cuda()
    pipeline.models["slat_flow_model"].cuda()
    print("Weights swapped.")

    # Replace samplers
    pipeline.sparse_structure_sampler = FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)
    pipeline.sparse_structure_sampler_params = {
        "steps": inf_params["ss_steps"],
        "cfg_strength": inf_params["ss_cfg"],
        "cfg_interval": inf_params.get("ss_cfg_interval", [0.5, 1.0]),
        "rescale_t": inf_params.get("rescale_t", 1.0),
    }
    pipeline.slat_sampler = FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)
    pipeline.slat_sampler_params = {
        "steps": inf_params["slat_steps"],
        "cfg_strength": inf_params["slat_cfg"],
        "cfg_interval": inf_params.get("slat_cfg_interval", [0.5, 1.0]),
        "rescale_t": inf_params.get("rescale_t", 1.0),
    }
    print(f"Samplers: {inf_params['ss_steps']} steps (ss) + "
          f"{inf_params['slat_steps']} steps (slat)")

    # Warmup
    if samples:
        warmup_s = samples[0]
        print(f"  [WARMUP] {warmup_s.object_id}")
        warmup_img = Image.open(warmup_s.input_image).convert("RGBA")
        with torch.no_grad():
            profile_one_sample(pipeline, warmup_img, args.seed)
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
                    pipeline, image, args.seed)
            t_total = sync_and_time() - t_total_start

            mesh_result = out["mesh"][0]
            peak_mem = torch.cuda.max_memory_allocated() / 1e9

            profile = {
                "object_id": s.object_id,
                "category": s.category,
                "model": "mdt_dist",
                "seed": args.seed,
                "blocks": {k: {"wall_sec": round(v, 4)} for k, v in block_timings.items()},
                "blocks_sum_sec": round(sum(block_timings.values()), 4),
                "total_sec": round(t_total, 4),
                "peak_gpu_gb": round(peak_mem, 2),
                "raw_vertices": int(mesh_result.vertices.shape[0]),
                "raw_faces": int(mesh_result.faces.shape[0]),
            }

            # Save mesh
            raw_verts = mesh_result.vertices.cpu().numpy()
            raw_faces = mesh_result.faces.cpu().numpy()
            trimesh.Trimesh(vertices=raw_verts, faces=raw_faces, process=False).export(
                os.path.join(out_dir, mesh_fn))

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
                "model": "mdt_dist",
                "status": "failed",
                "error": str(e),
            }
            with open(os.path.join(out_dir, "profile.json"), "w") as f:
                json.dump(profile, f, indent=2)


if __name__ == "__main__":
    main()
