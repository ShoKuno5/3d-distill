#!/usr/bin/env python3
"""Render 3-way comparison grids (input image | pred meshes | GT) using Blender.

For each resolution condition, produces a grid:
  rows = objects, cols = [input | trellis | trellis2 | hunyuan3d | hunyuan3d21 | GT]

Meshes are aligned to GT canonical space using the ICP transforms from evaluation.
Blender renders are parallelized across all available CPUs.

Usage:
    cd /mnt/workspace/kuno/distillation
    envs/hunyuan3d-venv/bin/python pipeline/scripts/render_grid_blender.py [--workers N]
"""

import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import trimesh
import yaml
from PIL import Image, ImageDraw, ImageFont

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR / "pipeline"))
from src.geometry.normalize import normalize_to_unit_sphere
from src.utils.inference_config import load_and_filter_samples, get_model_config

BLENDER = str(PROJECT_DIR / "envs" / "blender-2.80-linux-glibc217-x86_64" / "blender")
BLENDER_SCRIPT = str(PROJECT_DIR / "pipeline" / "scripts" / "_blender_render_mesh.py")

RESULTS_ROOT = PROJECT_DIR / "results" / "resolution_sweep"
CONDITIONS = ["res_300", "res_512", "res_1024"]
MODELS = ["trellis", "trellis2", "hunyuan3d", "hunyuan3d21"]
RENDER_RES = 512


def load_and_normalize_gt(mesh_path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(mesh_path, process=False, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    pts = mesh.vertices
    _, transform = normalize_to_unit_sphere(pts)
    mesh.vertices = (pts - transform.center) / transform.scale
    return mesh


def load_and_align_pred(pred_mesh_path: str, alignment_json_path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(pred_mesh_path, process=False, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    pts = mesh.vertices
    _, transform = normalize_to_unit_sphere(pts)
    mesh.vertices = (pts - transform.center) / transform.scale
    with open(alignment_json_path) as f:
        align_data = json.load(f)
    T = np.array(align_data["transform_4x4"])
    mesh.vertices = (T[:3, :3] @ mesh.vertices.T).T + T[:3, 3]
    return mesh


def render_one(obj_path: str, output_path: str, resolution: int = 512) -> tuple:
    """Render a single OBJ. Returns (output_path, success)."""
    cmd = [
        BLENDER, "--background", "--python", BLENDER_SCRIPT, "--",
        "--input", obj_path,
        "--output", output_path,
        "--resolution", str(resolution),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        return (output_path, result.returncode == 0)
    except Exception as e:
        return (output_path, False)


def load_input_image(path: str, resolution: int) -> Image.Image:
    img = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    composite = Image.alpha_composite(bg, img).convert("RGB")
    return composite.resize((resolution, resolution), Image.LANCZOS)


def prepare_render_jobs(condition, config_path, tmp_dir):
    """Export aligned OBJs and return list of (obj_path, png_path, grid_col, grid_row)."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    samples = load_and_filter_samples(cfg)
    output_root = cfg["output_root"]
    jobs = []

    for row, s in enumerate(samples):
        oid = s.object_id

        # Model predictions
        for col_off, model in enumerate(MODELS):
            col = col_off + 1
            norm_dir = os.path.join(output_root, "normalized_predictions", model, oid)
            pred_mesh_path = os.path.join(norm_dir, "mesh_cleaned.obj")
            align_json = os.path.join(norm_dir, "alignment_track_a.json")

            if not os.path.exists(pred_mesh_path) or not os.path.exists(align_json):
                continue

            try:
                aligned_mesh = load_and_align_pred(pred_mesh_path, align_json)
                tmp_obj = os.path.join(tmp_dir, f"{condition}_{model}_{oid}.obj")
                aligned_mesh.export(tmp_obj)
                tmp_png = os.path.join(tmp_dir, f"{condition}_{model}_{oid}.png")
                jobs.append((tmp_obj, tmp_png, col, row))
            except Exception as e:
                print(f"  Export error {model}/{oid}: {e}")

        # GT mesh
        try:
            gt_mesh = load_and_normalize_gt(s.mesh_obj)
            tmp_obj = os.path.join(tmp_dir, f"{condition}_gt_{oid}.obj")
            gt_mesh.export(tmp_obj)
            tmp_png = os.path.join(tmp_dir, f"{condition}_gt_{oid}.png")
            col_gt = len(MODELS) + 1
            jobs.append((tmp_obj, tmp_png, col_gt, row))
        except Exception as e:
            print(f"  GT export error {oid}: {e}")

    return jobs, samples


def assemble_grid(condition, samples, render_results, config_path):
    """Assemble rendered PNGs + input images into a grid."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    col_labels = ["Input"] + MODELS + ["GT"]
    n_cols = len(col_labels)
    n_rows = len(samples)

    grid = Image.new("RGB", (n_cols * RENDER_RES, n_rows * RENDER_RES), (240, 240, 240))

    # Input images
    for row, s in enumerate(samples):
        try:
            inp_img = load_input_image(s.input_image, RENDER_RES)
            grid.paste(inp_img, (0, row * RENDER_RES))
        except Exception as e:
            print(f"  Input error {s.object_id}: {e}")

    # Rendered meshes
    for png_path, col, row, ok in render_results:
        if ok and os.path.exists(png_path):
            cell = Image.open(png_path).convert("RGBA")
            # Composite transparent render onto white
            bg = Image.new("RGBA", cell.size, (255, 255, 255, 255))
            cell = Image.alpha_composite(bg, cell).convert("RGB")
            grid.paste(cell, (col * RENDER_RES, row * RENDER_RES))

    # Labels
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
        small_font = font

    for col, label in enumerate(col_labels):
        x = col * RENDER_RES + RENDER_RES // 2
        draw.text((x, 4), label, fill=(0, 0, 0), font=font, anchor="mt")

    for row, s in enumerate(samples):
        y = row * RENDER_RES + RENDER_RES - 6
        draw.text((6, y), s.object_id, fill=(80, 80, 80), font=small_font, anchor="lb")

    out_path = str(RESULTS_ROOT / f"{condition}_comparison.png")
    grid.save(out_path, quality=95)
    print(f"  Saved: {out_path} ({grid.size[0]}x{grid.size[1]}px)")
    return out_path


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16,
                        help="Parallel Blender processes")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="blender_render_") as tmp_dir:
        outputs = []

        for cond in CONDITIONS:
            config_path = str(PROJECT_DIR / "experiments" / "resolution_sweep"
                              / f"{cond}.yaml")
            if not os.path.exists(config_path):
                print(f"Config not found: {config_path}")
                continue

            print(f"\n{'='*60}")
            print(f"{cond}: Preparing meshes...")

            jobs, samples = prepare_render_jobs(cond, config_path, tmp_dir)
            print(f"{cond}: {len(jobs)} renders, {args.workers} workers")

            # Parallel Blender rendering
            render_results = []
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = {}
                for obj_path, png_path, col, row in jobs:
                    f = pool.submit(render_one, obj_path, png_path, RENDER_RES)
                    futures[f] = (png_path, col, row)

                done = 0
                for f in as_completed(futures):
                    png_path, col, row = futures[f]
                    out_path, ok = f.result()
                    render_results.append((png_path, col, row, ok))
                    done += 1
                    if done % 10 == 0 or done == len(jobs):
                        print(f"  {cond}: {done}/{len(jobs)} renders done")

            # Assemble grid
            out = assemble_grid(cond, samples, render_results, config_path)
            outputs.append(out)

        print(f"\n{'='*60}")
        print("All grids:")
        for p in outputs:
            print(f"  {p}")


if __name__ == "__main__":
    main()
