#!/usr/bin/env python3
"""Render ICP alignment comparison matrix.

For selected objects, renders a grid:
  cols = [Input | trellis2 raw | trellis2 ICP | hunyuan3d21 raw | hunyuan3d21 ICP | GT]
  rows = objects (sorted by max CD across models, worst first)

"raw" = unit-sphere normalized prediction (before ICP alignment)
"ICP" = after applying similarity_icp transform (aligned to GT)
"GT"  = unit-sphere normalized ground truth (alignment target)

Usage:
    cd /mnt/workspace/kuno/distillation
    envs/hunyuan3d-venv/bin/python experiments/category_pilot/render_icp_matrix.py
"""

import csv
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR / "pipeline"))
from src.geometry.normalize import normalize_to_unit_sphere

BLENDER = str(PROJECT_DIR / "envs" / "blender-3.6.16-linux-x64" / "blender")
BLENDER_SCRIPT = str(PROJECT_DIR / "pipeline" / "scripts" / "_blender_render_mesh.py")
RESULTS_ROOT = PROJECT_DIR / "results" / "category_pilot"
NORM_PRED_ROOT = RESULTS_ROOT / "normalized_predictions"
RENDER_RES = 512

MODELS = ["trellis2", "hunyuan3d21"]

# Selected objects covering diverse ICP behavior:
#   - guitar_017: both excellent
#   - chair_026: trellis2 great, h3d decent
#   - bottle_040: both good
#   - octopus_004: trellis2 good, h3d catastrophic (scale=0.5)
#   - mushroom_008: trellis2 mediocre, h3d catastrophic
#   - dog_093: trellis2 catastrophic (scale=0.5), h3d good
#   - toaster_013: trellis2 catastrophic, h3d decent
#   - plate_001: both bad (scale=0.5)
SELECTED_OBJECTS = [
    "guitar_017",
    "chair_026",
    "bottle_040",
    "octopus_004",
    "mushroom_008",
    "dog_093",
    "toaster_013",
    "plate_001",
]


def load_manifest():
    manifest = {}
    manifest_path = PROJECT_DIR / "experiments" / "category_pilot" / "manifest.csv"
    with open(manifest_path) as f:
        for row in csv.DictReader(f):
            manifest[row["object_id"]] = row
    return manifest


def load_metrics():
    """Load Track A metrics per (object_id, model)."""
    metrics = {}
    metrics_path = RESULTS_ROOT / "metrics" / "per_sample.csv"
    with open(metrics_path) as f:
        for row in csv.DictReader(f):
            if row["track"] == "A":
                key = (row["object_id"], row["model"])
                metrics[key] = {
                    "cd": float(row["chamfer_distance"]),
                    "scale": float(row["alignment_scale"]),
                    "rot_idx": int(row["alignment_rot_idx"]),
                }
    return metrics


def load_and_normalize_gt(mesh_path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(mesh_path, process=False, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    pts = mesh.vertices
    _, transform = normalize_to_unit_sphere(pts)
    mesh.vertices = (pts - transform.center) / transform.scale
    return mesh


def load_pred_raw(model: str, oid: str) -> trimesh.Trimesh:
    """Load normalized prediction mesh (before ICP)."""
    mesh_path = NORM_PRED_ROOT / model / oid / "mesh_cleaned.obj"
    mesh = trimesh.load(str(mesh_path), process=False, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    return mesh


def load_pred_icp(model: str, oid: str) -> trimesh.Trimesh:
    """Load normalized prediction mesh with ICP transform applied."""
    mesh = load_pred_raw(model, oid)
    align_path = NORM_PRED_ROOT / model / oid / "alignment_track_a.json"
    with open(align_path) as f:
        align_data = json.load(f)
    T = np.array(align_data["transform_4x4"])
    mesh.vertices = (T[:3, :3] @ mesh.vertices.T).T + T[:3, 3]
    return mesh


def decimate_if_needed(mesh: trimesh.Trimesh, max_faces: int = 100000) -> trimesh.Trimesh:
    if len(mesh.faces) > max_faces:
        mesh = mesh.simplify_quadric_decimation(face_count=max_faces)
    return mesh


def render_one(obj_path: str, output_path: str, resolution: int = 512) -> tuple:
    """Render a single OBJ via Blender. Returns (output_path, success)."""
    cmd = [
        BLENDER, "--background", "--python", BLENDER_SCRIPT, "--",
        "--input", obj_path,
        "--output", output_path,
        "--resolution", str(resolution),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return (output_path, result.returncode == 0)
    except Exception:
        return (output_path, False)


def load_input_image(path: str, resolution: int) -> Image.Image:
    img = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    composite = Image.alpha_composite(bg, img).convert("RGB")
    return composite.resize((resolution, resolution), Image.LANCZOS)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--objects", nargs="*", default=None,
                        help="Override default object selection")
    args = parser.parse_args()

    objects = args.objects or SELECTED_OBJECTS
    manifest = load_manifest()
    metrics = load_metrics()

    # Sort by worst-case CD (max across models), worst first
    def worst_cd(oid):
        cds = [metrics.get((oid, m), {}).get("cd", 0) for m in MODELS]
        return max(cds)
    objects = sorted(objects, key=worst_cd, reverse=True)

    # Column layout: Input | trellis2 raw | trellis2 ICP | hunyuan3d21 raw | hunyuan3d21 ICP | GT
    col_labels = ["Input", "trellis2\n(raw)", "trellis2\n(ICP)", "hunyuan3d21\n(raw)", "hunyuan3d21\n(ICP)", "GT"]
    n_cols = len(col_labels)
    n_rows = len(objects)

    with tempfile.TemporaryDirectory(prefix="icp_matrix_") as tmp_dir:
        # Export meshes and prepare render jobs: (obj_path, png_path, col, row)
        jobs = []
        print(f"Exporting {n_rows} objects x {n_cols - 1} mesh columns...")

        for row, oid in enumerate(objects):
            sample = manifest.get(oid)
            if not sample:
                print(f"  WARNING: {oid} not in manifest, skipping")
                continue

            # GT mesh (col 5)
            try:
                gt_mesh = load_and_normalize_gt(sample["mesh_obj"])
                gt_mesh = decimate_if_needed(gt_mesh)
                tmp_obj = os.path.join(tmp_dir, f"gt_{oid}.obj")
                gt_mesh.export(tmp_obj)
                tmp_png = os.path.join(tmp_dir, f"gt_{oid}.png")
                jobs.append((tmp_obj, tmp_png, 5, row))
            except Exception as e:
                print(f"  GT error {oid}: {e}")

            # Model predictions (raw + ICP)
            for model in MODELS:
                col_raw = 1 if model == "trellis2" else 3
                col_icp = 2 if model == "trellis2" else 4

                try:
                    # Raw (before ICP)
                    raw_mesh = load_pred_raw(model, oid)
                    raw_mesh = decimate_if_needed(raw_mesh)
                    tmp_obj = os.path.join(tmp_dir, f"{model}_raw_{oid}.obj")
                    raw_mesh.export(tmp_obj)
                    tmp_png = os.path.join(tmp_dir, f"{model}_raw_{oid}.png")
                    jobs.append((tmp_obj, tmp_png, col_raw, row))

                    # ICP aligned
                    icp_mesh = load_pred_icp(model, oid)
                    icp_mesh = decimate_if_needed(icp_mesh)
                    tmp_obj = os.path.join(tmp_dir, f"{model}_icp_{oid}.obj")
                    icp_mesh.export(tmp_obj)
                    tmp_png = os.path.join(tmp_dir, f"{model}_icp_{oid}.png")
                    jobs.append((tmp_obj, tmp_png, col_icp, row))
                except Exception as e:
                    print(f"  {model} error {oid}: {e}")

        print(f"Rendering {len(jobs)} meshes with {args.workers} Blender workers...")

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
                    print(f"  {done}/{len(jobs)} renders done")

        # Assemble grid
        label_height = 50
        info_width = 180  # Right-side annotation column
        grid_w = n_cols * RENDER_RES + info_width
        grid_h = label_height + n_rows * RENDER_RES
        grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))

        # Paste input images (col 0)
        for row, oid in enumerate(objects):
            sample = manifest.get(oid)
            if sample:
                try:
                    inp_img = load_input_image(sample["input_image"], RENDER_RES)
                    grid.paste(inp_img, (0, label_height + row * RENDER_RES))
                except Exception as e:
                    print(f"  Input error {oid}: {e}")

        # Paste rendered meshes
        for png_path, col, row, ok in render_results:
            if ok and os.path.exists(png_path):
                cell = Image.open(png_path).convert("RGBA")
                bg = Image.new("RGBA", cell.size, (255, 255, 255, 255))
                cell = Image.alpha_composite(bg, cell).convert("RGB")
                grid.paste(cell, (col * RENDER_RES, label_height + row * RENDER_RES))

        # Draw labels and annotations
        draw = ImageDraw.Draw(grid)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
            tiny = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        except OSError:
            font = ImageFont.load_default()
            small = font
            tiny = font

        # Column headers
        for col, label in enumerate(col_labels):
            x = col * RENDER_RES + RENDER_RES // 2
            lines = label.split("\n")
            for i, line in enumerate(lines):
                draw.text((x, 4 + i * 18), line, fill=(0, 0, 0), font=font, anchor="mt")

        # Row labels + metrics annotation on right side
        for row, oid in enumerate(objects):
            y_base = label_height + row * RENDER_RES
            # Object ID on left edge
            draw.text((6, y_base + RENDER_RES - 6), oid, fill=(60, 60, 60),
                       font=small, anchor="lb")

            # Metrics annotation on right
            x_info = n_cols * RENDER_RES + 4
            y_info = y_base + 8
            draw.text((x_info, y_info), oid, fill=(0, 0, 0), font=small)
            y_info += 18

            for model in MODELS:
                m = metrics.get((oid, model), {})
                cd = m.get("cd", 0) * 1000
                scale = m.get("scale", 0)
                label_str = f"{model[:4]}2: CD={cd:.1f}"
                color = (200, 0, 0) if cd > 10 else (0, 100, 0)
                draw.text((x_info, y_info), label_str, fill=color, font=tiny)
                y_info += 14
                draw.text((x_info, y_info), f"  s={scale:.3f}", fill=(80, 80, 80), font=tiny)
                y_info += 14

        # Separator lines between trellis2 and hunyuan3d21 columns
        sep_x = 3 * RENDER_RES
        draw.line([(sep_x, 0), (sep_x, grid_h)], fill=(180, 180, 180), width=2)
        # Separator before GT
        sep_x2 = 5 * RENDER_RES
        draw.line([(sep_x2, 0), (sep_x2, grid_h)], fill=(180, 180, 180), width=2)

        # Row separators
        for row in range(1, n_rows):
            y = label_height + row * RENDER_RES
            draw.line([(0, y), (grid_w, y)], fill=(220, 220, 220), width=1)

        out_dir = RESULTS_ROOT / "qualitative"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "icp_alignment_matrix.png"
        grid.save(str(out_path), quality=95)
        print(f"\nSaved: {out_path} ({grid.size[0]}x{grid.size[1]}px)")
        print(f"Objects (top=worst): {objects}")


if __name__ == "__main__":
    main()
