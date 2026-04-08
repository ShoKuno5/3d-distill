#!/usr/bin/env python3
"""Decode teacher ODE intermediate latents to visualize structure formation.

Runs the teacher's 50-step Euler ODE and decodes the latent at specified
intermediate steps into meshes. Records mesh statistics and Chamfer distance
to GT at each step. Saves latents (.npy) for later analysis.

Usage:
    cd /mnt/workspace/kuno/distillation/models/hunyuan3d21/hy3dshape
    PYTHONPATH=.:../../../distill_methods/src:../../../pipeline \
      CUDA_VISIBLE_DEVICES=0 \
      ../../../envs/hunyuan3d-venv/bin/python \
      ../../../distill_methods/scripts/decode_intermediate_steps.py \
      --config ../../../distill_methods/config.yaml --seed 42
"""

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import trimesh
import yaml
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_DIR / "pipeline"))
sys.path.insert(0, str(PROJECT_DIR / "models" / "hunyuan3d21" / "hy3dshape"))

from src.utils.inference_config import load_and_filter_samples
from src.geometry.normalize import normalize_to_unit_sphere
from src.geometry.cleaning import clean_mesh
from src.geometry.sampling import sample_surface
from src.evaluation.metrics import compute_geometry_metrics

# Categories selected for shape complexity diversity
SELECTED_CATEGORIES = [
    "ball",        # simplest (sphere)
    "apple",       # simple (convex)
    "cup",         # concavity
    "shoe",        # asymmetric
    "chair",       # structural
    "guitar",      # elongated
    "airplane",    # complex (wings)
    "robot",       # complex (articulated)
    "dinosaur",    # organic complex
    "helicopter",  # most complex (rotor)
]

# Steps at which to decode (out of 50)
DECODE_STEPS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]


def get_latent_shape(teacher):
    """Determine the latent shape the DiT expects."""
    if hasattr(teacher, "num_latents") and hasattr(teacher, "in_channels"):
        return (teacher.num_latents, teacher.in_channels)
    if hasattr(teacher, "x_embedder"):
        proj = teacher.x_embedder
        if hasattr(proj, "in_features"):
            latent_dim = proj.in_features
        elif hasattr(proj, "weight"):
            latent_dim = proj.weight.shape[1]
        else:
            latent_dim = 64
        return (4096, latent_dim)
    return (4096, 64)


def teacher_ode_solve_with_snapshots(
    model, noise, contexts, num_steps, guidance_scale, snapshot_steps,
):
    """Run Euler ODE from t=0 (noise) to t=1 (data), saving snapshots.

    Returns dict mapping step_number -> latent tensor (on GPU).
    """
    dt = 1.0 / num_steps
    x_t = noise.clone()
    snapshots = {}

    for i in range(num_steps):
        t = torch.full((noise.shape[0],), i * dt, device=noise.device)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            v_cond = model(x_t, t, contexts=contexts)
            if guidance_scale > 1.0:
                null_ctx = {k: torch.zeros_like(v) for k, v in contexts.items()}
                v_uncond = model(x_t, t, contexts=null_ctx)
                v = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                v = v_cond

        x_t = x_t + dt * v

        step_num = i + 1
        if step_num in snapshot_steps:
            snapshots[step_num] = x_t.clone()

    return snapshots


def decode_latent(pipeline, latent, octree_resolution):
    """Decode a latent tensor into a trimesh mesh via the pipeline's VAE."""
    from hy3dshape.pipelines import export_to_trimesh

    # Match VAE weight dtype (typically fp16)
    vae_dtype = next(pipeline.vae.parameters()).dtype
    latent = latent.to(dtype=vae_dtype)

    scaled = 1.0 / pipeline.vae.scale_factor * latent
    decoded = pipeline.vae(scaled)
    outputs = pipeline.vae.latents2mesh(
        decoded,
        bounds=1.01,
        mc_level=0.0,
        num_chunks=8000,
        octree_resolution=octree_resolution,
        mc_algo="mc",
        enable_pbar=False,
    )
    meshes = export_to_trimesh(outputs)
    return meshes[0] if meshes else None


def compute_mesh_stats(mesh):
    """Basic mesh statistics."""
    if mesh is None:
        return {"vertex_count": 0, "face_count": 0, "bbox_volume": 0.0}
    bbox = mesh.bounding_box.extents
    return {
        "vertex_count": len(mesh.vertices),
        "face_count": len(mesh.faces),
        "bbox_volume": float(np.prod(bbox)),
    }


def process_gt_mesh(mesh_path, clean_cfg, n_points):
    """Load, clean, sample, and normalize GT mesh."""
    mesh = trimesh.load(mesh_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    mesh, report = clean_mesh(
        mesh,
        remove_nan=clean_cfg["remove_nan"],
        remove_inf=clean_cfg["remove_inf"],
        remove_degenerate=clean_cfg["remove_degenerate_faces"],
        remove_unreferenced=clean_cfg["remove_unreferenced_vertices"],
        remove_tiny_components=clean_cfg["remove_tiny_components"],
        tiny_threshold=clean_cfg["tiny_component_threshold"],
    )
    if report.is_empty:
        return None
    pts = sample_surface(mesh, n_points, seed=0)
    pts_norm, _ = normalize_to_unit_sphere(pts)
    return pts_norm


def process_pred_mesh(mesh, clean_cfg, n_points):
    """Clean, sample, and normalize a predicted mesh."""
    if mesh is None:
        return None
    mesh, report = clean_mesh(
        mesh,
        remove_nan=clean_cfg["remove_nan"],
        remove_inf=clean_cfg["remove_inf"],
        remove_degenerate=clean_cfg["remove_degenerate_faces"],
        remove_unreferenced=clean_cfg["remove_unreferenced_vertices"],
        remove_tiny_components=clean_cfg["remove_tiny_components"],
        tiny_threshold=clean_cfg["tiny_component_threshold"],
    )
    if report.is_empty:
        return None
    pts = sample_surface(mesh, n_points, seed=0)
    pts_norm, _ = normalize_to_unit_sphere(pts)
    return pts_norm


def main():
    parser = argparse.ArgumentParser(
        description="Decode teacher intermediate ODE steps"
    )
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-samples", type=int, default=10,
        help="Number of samples (selects from SELECTED_CATEGORIES)",
    )
    parser.add_argument("--octree-resolution", type=int, default=384)
    parser.add_argument(
        "--guidance-scale", type=float, default=None,
        help="CFG scale (default: from models.teacher_50step.inference_params, typically 5.0)",
    )
    parser.add_argument(
        "--output-suffix", type=str, default="",
        help="Suffix for output directory (e.g. '_cfg5' or '_cfg7.5')",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Load test samples and filter by selected categories
    all_samples = load_and_filter_samples(
        cfg, manifest_key="test_manifest",
    )
    sample_map = {s.category: s for s in all_samples}
    samples = []
    for cat in SELECTED_CATEGORIES[: args.num_samples]:
        if cat in sample_map:
            samples.append(sample_map[cat])
        else:
            print(f"WARNING: category '{cat}' not found in test manifest")

    if not samples:
        print("ERROR: No matching samples found")
        sys.exit(1)

    print(f"Selected {len(samples)} samples: {[s.object_id for s in samples]}")
    print(f"Decode steps: {DECODE_STEPS}")

    out_root = os.path.join(cfg["output_root"], f"intermediate_decode{args.output_suffix}")
    os.makedirs(out_root, exist_ok=True)

    clean_cfg = cfg["cleaning"]
    n_eval_points = cfg["sampling"]["eval_points"]
    num_steps = 50

    # Resolve guidance_scale: CLI > teacher inference params > training config
    if args.guidance_scale is not None:
        guidance_scale = args.guidance_scale
    else:
        # Use teacher's actual inference guidance_scale (5.0), not training CFG (7.5)
        teacher_cfg = next(
            (m for m in cfg.get("models", []) if m["name"] == "teacher_50step"), None
        )
        if teacher_cfg:
            guidance_scale = teacher_cfg["inference_params"].get("guidance_scale", 5.0)
        else:
            guidance_scale = 5.0
    print(f"Guidance scale: {guidance_scale}")

    # Load pipeline (need VAE on GPU for decode)
    print("Loading Hunyuan3D-2.1 pipeline ...")
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        cfg["training"]["model"]["pretrained"],
        use_safetensors=False,
    )

    teacher = pipeline.model
    cond_model = pipeline.conditioner
    device = torch.device("cuda")

    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    cond_model.to(device)
    # Keep VAE on GPU for decode
    pipeline.vae.to(device)

    latent_shape = get_latent_shape(teacher)
    print(f"Latent shape: {latent_shape}")

    summary_rows = []

    for si, s in enumerate(samples):
        print(f"\n[{si+1}/{len(samples)}] {s.object_id} ({s.category})")
        sample_dir = os.path.join(out_root, s.object_id)
        os.makedirs(sample_dir, exist_ok=True)

        # Skip if already completed (stats.json exists with no dtype errors)
        stats_path = os.path.join(sample_dir, "stats.json")
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                existing_stats = json.load(f)
            has_dtype_error = any("dtype" in (r.get("error") or "") for r in existing_stats)
            if not has_dtype_error and len(existing_stats) == len(DECODE_STEPS):
                print(f"  Already completed, loading existing results.")
                summary_rows.extend(existing_stats)
                continue

        # Prepare GT points
        gt_pts = None
        if hasattr(s, "mesh_obj") and s.mesh_obj and os.path.exists(s.mesh_obj):
            try:
                gt_pts = process_gt_mesh(s.mesh_obj, clean_cfg, n_eval_points)
                print(f"  GT mesh loaded: {s.mesh_obj}")
            except Exception as e:
                print(f"  WARNING: GT mesh failed: {e}")

        # Encode condition
        image = Image.open(s.input_image).convert("RGBA")
        cond_input = pipeline.prepare_image(image)
        img_tensor = cond_input["image"].to(device)
        mask_tensor = cond_input["mask"].to(device)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            contexts = cond_model(image=img_tensor, mask=mask_tensor)

        # Generate noise (deterministic)
        rng = torch.Generator(device=device).manual_seed(args.seed)
        noise = torch.randn(1, *latent_shape, device=device, generator=rng)

        # Run ODE with snapshots
        print(f"  Running {num_steps}-step ODE ...")
        t_ode_start = time.time()
        snapshots = teacher_ode_solve_with_snapshots(
            teacher, noise, contexts,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            snapshot_steps=set(DECODE_STEPS),
        )
        t_ode = time.time() - t_ode_start
        print(f"  ODE done: {t_ode:.1f}s")

        sample_stats = []

        for step_num in DECODE_STEPS:
            latent = snapshots[step_num]
            step_label = f"step_{step_num:02d}"

            # Save latent
            latent_np = latent.float().cpu().numpy()[0]
            np.save(os.path.join(sample_dir, f"latent_{step_label}.npy"), latent_np)

            # Decode to mesh
            result = {
                "object_id": s.object_id,
                "category": s.category,
                "step": step_num,
                "t": step_num / num_steps,
                "status": "success",
                "error": None,
            }

            try:
                t_dec_start = time.time()
                mesh = decode_latent(pipeline, latent, args.octree_resolution)
                t_dec = time.time() - t_dec_start

                stats = compute_mesh_stats(mesh)
                result.update(stats)
                result["decode_time_sec"] = round(t_dec, 2)

                # Save mesh
                if mesh is not None and len(mesh.vertices) > 0:
                    mesh.export(os.path.join(sample_dir, f"{step_label}.obj"))

                    # Compute CD against GT
                    if gt_pts is not None:
                        pred_pts = process_pred_mesh(
                            mesh, clean_cfg, n_eval_points,
                        )
                        if pred_pts is not None:
                            metrics = compute_geometry_metrics(pred_pts, gt_pts)
                            result.update(asdict(metrics))
                        else:
                            result["status"] = "empty_after_clean"
                else:
                    result["status"] = "empty_mesh"

            except Exception as e:
                result["status"] = "failed"
                result["error"] = str(e)
                result["decode_time_sec"] = 0.0

            sample_stats.append(result)

            status_str = result["status"]
            cd_str = ""
            if "chamfer_distance" in result and result["chamfer_distance"] is not None:
                cd_str = f", CD={result['chamfer_distance']:.6f}"
            verts = result.get("vertex_count", 0)
            print(f"    {step_label} (t={result['t']:.2f}): {status_str}, "
                  f"verts={verts}{cd_str}")

            summary_rows.append(result)

        # Save per-sample stats
        with open(os.path.join(sample_dir, "stats.json"), "w") as f:
            json.dump(sample_stats, f, indent=2, default=str)

    # Write summary CSV
    if summary_rows:
        csv_path = os.path.join(out_root, "summary.csv")
        fieldnames = list(summary_rows[0].keys())
        # Ensure all keys from all rows are included
        for row in summary_rows:
            for k in row:
                if k not in fieldnames:
                    fieldnames.append(k)

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\nSummary written to {csv_path}")

    print("Done.")


if __name__ == "__main__":
    main()
