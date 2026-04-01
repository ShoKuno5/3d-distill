#!/usr/bin/env python3
"""Compare teacher (50-step) vs PD student (25-step) on same input.

Runs both, exports meshes, computes Chamfer Distance.

Usage:
    cd /mnt/workspace/kuno/distillation/models/hunyuan3d21/hy3dshape
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 ../../../envs/hunyuan3d-venv/bin/python -u \
        ../../../distill_methods/scripts/compare_teacher_student.py \
        --lora-path ../../../results/distill_methods/checkpoints/pd/step_4000 \
        --image ../../../datasets/Toys4k/renders/512/airplane/airplane_006/image.png \
        --output-dir /tmp/pd_comparison
"""

import argparse
import os
import time

import numpy as np
import torch
import trimesh
from PIL import Image


def sample_points(mesh, n=100000):
    """Sample points uniformly from mesh surface."""
    points, _ = trimesh.sample.sample_surface(mesh, n)
    return points


def chamfer_distance(p1, p2):
    """Chamfer Distance (L2) between two point clouds."""
    from scipy.spatial import cKDTree
    tree1 = cKDTree(p1)
    tree2 = cKDTree(p2)
    d1, _ = tree1.query(p2)
    d2, _ = tree2.query(p1)
    return float(np.mean(d1**2) + np.mean(d2**2))


def run_inference(pipeline, image, num_steps, guidance_scale, octree_resolution, seed):
    """Run pipeline inference and return mesh + timing."""
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    mesh = pipeline(
        image=image,
        generator=torch.manual_seed(seed),
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        octree_resolution=octree_resolution,
        output_type="trimesh",
    )[0]
    runtime = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    return mesh, runtime, peak_mem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora-path", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", default="/tmp/pd_comparison")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-points", type=int, default=100000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    image = Image.open(args.image).convert("RGBA")
    print(f"Input: {args.image}")

    # Load pipeline
    print("Loading pipeline ...")
    t0 = time.time()
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2.1", use_safetensors=False,
    )
    print(f"Pipeline loaded in {time.time() - t0:.1f}s")

    # --- Teacher: 50 steps, CFG=5.0, no LoRA ---
    print("\n=== Teacher: 50 steps, guidance_scale=5.0 ===")
    teacher_mesh, t_time, t_mem = run_inference(
        pipeline, image, num_steps=50, guidance_scale=5.0,
        octree_resolution=384, seed=args.seed,
    )
    teacher_path = os.path.join(args.output_dir, "teacher_50step.obj")
    teacher_mesh.export(teacher_path)
    print(f"  Time: {t_time:.1f}s, Peak GPU: {t_mem:.1f}GB")
    print(f"  Vertices: {len(teacher_mesh.vertices)}, Faces: {len(teacher_mesh.faces)}")
    print(f"  Exported: {teacher_path}")

    # --- Student: apply LoRA, 25 steps, CFG=1.0 ---
    print(f"\nLoading LoRA from {args.lora_path} ...")
    from peft import PeftModel
    dit = pipeline.model
    pipeline.model = PeftModel.from_pretrained(dit, args.lora_path)
    pipeline.model = pipeline.model.merge_and_unload()
    print("LoRA merged.")

    print("\n=== Student (PD step_4000): 25 steps, guidance_scale=1.0 ===")
    student_mesh, s_time, s_mem = run_inference(
        pipeline, image, num_steps=25, guidance_scale=1.0,
        octree_resolution=384, seed=args.seed,
    )
    student_path = os.path.join(args.output_dir, "student_pd_25step.obj")
    student_mesh.export(student_path)
    print(f"  Time: {s_time:.1f}s, Peak GPU: {s_mem:.1f}GB")
    print(f"  Vertices: {len(student_mesh.vertices)}, Faces: {len(student_mesh.faces)}")
    print(f"  Exported: {student_path}")

    # --- Compare ---
    print(f"\nSampling {args.sample_points} points from each mesh ...")
    pts_teacher = sample_points(teacher_mesh, args.sample_points)
    pts_student = sample_points(student_mesh, args.sample_points)

    # Normalize both to unit sphere for fair comparison
    for name, pts in [("teacher", pts_teacher), ("student", pts_student)]:
        centroid = pts.mean(axis=0)
        pts -= centroid
        scale = np.max(np.linalg.norm(pts, axis=1))
        pts /= scale
        if name == "teacher":
            pts_teacher = pts
        else:
            pts_student = pts

    cd = chamfer_distance(pts_teacher, pts_student)
    print(f"\n{'='*50}")
    print(f"Chamfer Distance (normalized): {cd:.6f}")
    print(f"Teacher: {len(teacher_mesh.vertices)} verts, {t_time:.1f}s")
    print(f"Student: {len(student_mesh.vertices)} verts, {s_time:.1f}s")
    print(f"Speedup: {t_time/s_time:.1f}x")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
