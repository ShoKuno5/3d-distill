#!/usr/bin/env python3
"""Quick inference test: load PD LoRA checkpoint, run 25-step inference, export mesh.

Usage:
    cd /mnt/workspace/kuno/distillation/models/hunyuan3d21/hy3dshape
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 ../../../envs/hunyuan3d-venv/bin/python \
        ../../../distill_methods/scripts/test_inference_pd.py \
        --lora-path ../../../results/distill_methods/checkpoints/pd/step_4000 \
        --num-steps 25 \
        --image ../../../datasets/Toys4k/renders/512/airplane/airplane_006/image.png \
        --output /tmp/pd_test_inference.obj
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora-path", required=True, help="Path to LoRA adapter dir")
    parser.add_argument("--num-steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--octree-resolution", type=int, default=384)
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output mesh path (.obj)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", default="tencent/Hunyuan3D-2.1",
                        help="Base model path")
    args = parser.parse_args()

    print(f"Loading pipeline from {args.pretrained} ...")
    t0 = time.time()
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        args.pretrained, use_safetensors=False,
    )
    print(f"Pipeline loaded in {time.time() - t0:.1f}s")

    print(f"Loading LoRA from {args.lora_path} ...")
    from peft import PeftModel
    dit = pipeline.model
    pipeline.model = PeftModel.from_pretrained(dit, args.lora_path)
    pipeline.model = pipeline.model.merge_and_unload()
    print("LoRA merged.")

    image = Image.open(args.image).convert("RGBA")
    print(f"Input image: {args.image} ({image.size})")

    print(f"Running inference: {args.num_steps} steps, guidance_scale={args.guidance_scale} ...")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    mesh = pipeline(
        image=image,
        generator=torch.manual_seed(args.seed),
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        octree_resolution=args.octree_resolution,
        output_type="trimesh",
    )[0]

    runtime = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"Inference done: {runtime:.1f}s, peak GPU {peak_mem:.1f}GB")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    mesh.export(args.output)
    print(f"Mesh exported: {args.output}")

    # Basic mesh stats
    print(f"  Vertices: {len(mesh.vertices)}")
    print(f"  Faces: {len(mesh.faces)}")
    bbox = mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)
    print(f"  Bounding box: {bbox}")


if __name__ == "__main__":
    main()
