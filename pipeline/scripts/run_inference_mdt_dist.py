#!/usr/bin/env python3
"""MDT-dist inference wrapper for aligned evaluation.

Loads TRELLIS v1 pipeline, swaps flow model weights with MDT-dist distilled
checkpoints, and runs 2-step inference.

Usage (from repo root):
    cd models/trellis && PYTHONPATH=. \
    ../../envs/miniconda3/envs/trellis/bin/python \
    ../../pipeline/scripts/run_inference_mdt_dist.py \
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
    parser.add_argument("--check-only", action="store_true",
                        help="Only check which outputs exist, don't run inference")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Intra-node sharding via env (samples sliced as samples[shard::num_shards]).
    shard_index = int(os.environ.get("MDT_SHARD_INDEX", "0"))
    num_shards = int(os.environ.get("MDT_NUM_SHARDS", "1"))
    if num_shards < 1 or not (0 <= shard_index < num_shards):
        raise SystemExit(f"invalid sharding: shard_index={shard_index}, num_shards={num_shards}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    samples = load_and_filter_samples(cfg, max_samples_override=args.max_samples, manifest_key="test_manifest")
    if num_shards > 1:
        before = len(samples)
        samples = samples[shard_index::num_shards]
        print(f"Shard {shard_index}/{num_shards}: {len(samples)}/{before} samples")

    model_cfg = get_model_config(cfg, "mdt_dist")
    if model_cfg is None:
        print("ERROR: 'mdt_dist' not found in config models list")
        sys.exit(1)
    pred_root = model_cfg["predictions_root"]
    mesh_fn = model_cfg["mesh_filename"]

    inf_params = get_inference_params(cfg, "mdt_dist")
    print(f"MDT-dist inference params: {inf_params}")

    existing = 0
    missing = []
    for s in samples:
        mesh_path = os.path.join(pred_root, s.object_id, mesh_fn)
        if os.path.exists(mesh_path):
            existing += 1
        else:
            missing.append(s)

    print(f"MDT-dist: {existing}/{len(samples)} predictions exist")

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
        os.environ['SPCONV_ALGO'] = 'native'
        from trellis.pipelines import TrellisImageTo3DPipeline
        from trellis.pipelines.samplers import FlowEulerGuidanceIntervalSampler
        from PIL import Image
        import trimesh
        import json
        import time
    except ImportError as e:
        print(f"Cannot import TRELLIS: {e}")
        print("Run from models/trellis directory with proper PYTHONPATH.")
        sys.exit(1)

    # Step 1: Load original TRELLIS v1 pipeline
    print("Loading TRELLIS v1 pipeline...")
    pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipeline.cuda()

    # Step 2: Swap flow model weights with MDT-dist checkpoints
    mdt_ckpts = inf_params.get("ckpts_dir",
        str(Path(__file__).resolve().parents[2] / "models" / "mdt_dist" / "ckpts"))
    print(f"Swapping weights from {mdt_ckpts}...")

    ss_flow_path = os.path.join(mdt_ckpts, "ss_flow_img_dit_L_16l8_fp16.pt")
    slat_flow_path = os.path.join(mdt_ckpts, "slat_flow_img_dit_L_64l8p2_fp16.pt")

    ss_flow_state = torch.load(ss_flow_path, map_location="cpu", weights_only=True)
    slat_flow_state = torch.load(slat_flow_path, map_location="cpu", weights_only=True)

    pipeline.models["sparse_structure_flow_model"].load_state_dict(ss_flow_state)
    pipeline.models["slat_flow_model"].load_state_dict(slat_flow_state)
    pipeline.models["sparse_structure_flow_model"].cuda()
    pipeline.models["slat_flow_model"].cuda()
    print("Weights swapped successfully.")

    # Step 3: Replace samplers with 2-step distilled configuration
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
    print(f"Samplers set to {inf_params['ss_steps']} steps (ss) + {inf_params['slat_steps']} steps (slat).")

    # Step 4: Run inference
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
                formats=["mesh"],
            )

            mesh_result = outputs["mesh"][0]
            raw_verts = mesh_result.vertices.cpu().numpy()
            raw_faces = mesh_result.faces.cpu().numpy()
            raw_mesh = trimesh.Trimesh(vertices=raw_verts, faces=raw_faces, process=False)
            raw_mesh.export(os.path.join(out_dir, mesh_fn))

            runtime = time.time() - t0
            peak_mem = torch.cuda.max_memory_allocated() / 1e9

            meta = {
                "model": "mdt_dist", "inference_params": inf_params,
                "object_id": s.object_id, "category": s.category,
                "input_image": s.input_image,
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
