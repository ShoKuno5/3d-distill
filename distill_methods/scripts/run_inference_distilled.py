#!/usr/bin/env python3
"""Inference wrapper for distilled Hunyuan3D-2.1 models.

Loads the base pipeline, applies LoRA weights, and runs inference at
the configured number of steps. Supports PD, CD, DMD1, DMD2 checkpoints.

Usage:
    cd /mnt/workspace/kuno/distillation/models/hunyuan3d21
    PYTHONPATH=. ../../envs/hunyuan3d-venv/bin/python \
        ../../distill_methods/scripts/run_inference_distilled.py \
        --config ../../distill_methods/config.yaml \
        --model-name pd_6step
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_DIR / "pipeline"))

from src.utils.inference_config import get_model_config, get_inference_params, load_and_filter_samples


def main():
    parser = argparse.ArgumentParser(description="Distilled model inference")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-name", required=True,
                        help="Model name from config (e.g. pd_6step, cd_4step, dmd1_1step, dmd2_1step)")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    samples = load_and_filter_samples(cfg, max_samples_override=args.max_samples, manifest_key="test_manifest")
    model_cfg = get_model_config(cfg, args.model_name)
    if model_cfg is None:
        print(f"ERROR: '{args.model_name}' not found in config models list")
        sys.exit(1)

    pred_root = model_cfg["predictions_root"]
    mesh_fn = model_cfg["mesh_filename"]
    inf_params = get_inference_params(cfg, args.model_name)
    print(f"{args.model_name} inference params: {inf_params}")

    # Check existing predictions
    existing = 0
    missing = []
    for s in samples:
        mesh_path = os.path.join(pred_root, s.object_id, mesh_fn)
        if os.path.exists(mesh_path):
            existing += 1
        else:
            missing.append(s)

    print(f"{args.model_name}: {existing}/{len(samples)} predictions exist")

    if args.check_only:
        if missing:
            print(f"Missing: {[s.object_id for s in missing]}")
        return

    if not missing:
        print("All predictions available. Nothing to do.")
        return

    print(f"Running inference for {len(missing)} missing samples ...")

    import torch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    from PIL import Image

    # Load base pipeline
    print("Loading Hunyuan3D-2.1 pipeline ...")
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        cfg["training"]["model"]["pretrained"],
        use_safetensors=False,
    )

    # Apply LoRA weights if specified
    lora_path = inf_params.get("lora_path")
    if lora_path:
        print(f"Loading LoRA from {lora_path} ...")
        from peft import PeftModel

        dit = pipeline.model
        pipeline.model = PeftModel.from_pretrained(dit, lora_path)
        pipeline.model = pipeline.model.merge_and_unload()
        print("LoRA merged into base model.")

    num_steps = inf_params.get("num_inference_steps", 50)
    guidance_scale = inf_params.get("guidance_scale", 1.0)
    octree_resolution = inf_params.get("octree_resolution", 384)

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
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                octree_resolution=octree_resolution,
                output_type="trimesh",
            )[0]
            mesh.export(os.path.join(out_dir, mesh_fn))

            runtime = time.time() - t0
            peak_mem = torch.cuda.max_memory_allocated() / 1e9

            meta = {
                "model": args.model_name,
                "object_id": s.object_id,
                "category": s.category,
                "input_image": s.input_image,
                "inference_params": {
                    "num_inference_steps": num_steps,
                    "guidance_scale": guidance_scale,
                    "octree_resolution": octree_resolution,
                    "lora_path": lora_path,
                },
                "runtime_sec": round(runtime, 2),
                "peak_gpu_gb": round(peak_mem, 2),
                "seed": args.seed,
                "status": "success",
            }
            with open(os.path.join(out_dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
            print(f"    Done: {runtime:.1f}s, {peak_mem:.1f}GB")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"    FAILED: {e}")


if __name__ == "__main__":
    main()
