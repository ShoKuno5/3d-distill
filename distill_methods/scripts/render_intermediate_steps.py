#!/usr/bin/env python3
"""Render intermediate decode meshes into per-sample step grids.

For each sample, renders all available step meshes and composites them
into a single row image showing structure formation over ODE steps.

Usage:
    cd /mnt/workspace/kuno/distillation
    python distill_methods/scripts/render_intermediate_steps.py

Uses Blender 3.6 with CYCLES GPU rendering.
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
BLENDER = str(PROJECT_DIR / "envs" / "blender-3.6.16-linux-x64" / "blender")
BLENDER_SCRIPT = str(PROJECT_DIR / "pipeline" / "scripts" / "_blender_render_mesh.py")
DECODE_ROOT = PROJECT_DIR / "results" / "distill_methods" / "intermediate_decode"

RENDER_RES = 512
STEPS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]


def render_one(obj_path: str, output_path: str, resolution: int) -> tuple:
    """Render a single OBJ via Blender. Returns (output_path, success)."""
    cmd = [
        BLENDER, "--background", "--python", BLENDER_SCRIPT, "--",
        "--input", obj_path,
        "--output", output_path,
        "--resolution", str(resolution),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
        )
        return (output_path, result.returncode == 0, result.stderr)
    except Exception as e:
        return (output_path, False, str(e))


def make_grid(sample_dir: str, render_dir: str, sample_id: str) -> str | None:
    """Composite step renders into a single row grid image."""
    images = []
    labels = []
    for step in STEPS:
        png_path = os.path.join(render_dir, f"step_{step:02d}.png")
        if os.path.exists(png_path):
            img = Image.open(png_path).convert("RGBA")
            # Composite onto white background
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            bg.paste(img, mask=img)
            images.append(bg.convert("RGB"))
            labels.append(f"t={step/50:.1f}")
        else:
            # Empty placeholder
            placeholder = Image.new("RGB", (RENDER_RES, RENDER_RES), (240, 240, 240))
            draw = ImageDraw.Draw(placeholder)
            text = f"t={step/50:.1f}\n(empty)"
            bbox = draw.textbbox((0, 0), text)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text(
                ((RENDER_RES - tw) // 2, (RENDER_RES - th) // 2),
                text, fill=(180, 180, 180),
            )
            images.append(placeholder)
            labels.append(f"t={step/50:.1f}")

    if not images:
        return None

    # Create grid: 1 row of N images with labels
    label_height = 30
    cell_w = RENDER_RES
    cell_h = RENDER_RES + label_height
    grid_w = cell_w * len(images)
    grid_h = cell_h

    grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(grid)

    for i, (img, label) in enumerate(zip(images, labels)):
        x = i * cell_w
        grid.paste(img, (x, label_height))
        bbox = draw.textbbox((0, 0), label)
        tw = bbox[2] - bbox[0]
        draw.text(
            (x + (cell_w - tw) // 2, 5),
            label, fill=(0, 0, 0),
        )

    grid_path = os.path.join(render_dir, f"{sample_id}_grid.png")
    grid.save(grid_path)
    return grid_path


def main():
    global RENDER_RES

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=RENDER_RES)
    args = parser.parse_args()

    RENDER_RES = args.resolution

    # Collect all OBJ files to render
    render_jobs = []
    sample_dirs = sorted(DECODE_ROOT.iterdir())
    for sample_dir in sample_dirs:
        if not sample_dir.is_dir():
            continue
        render_dir = sample_dir / "renders"
        render_dir.mkdir(exist_ok=True)
        for step in STEPS:
            obj_path = sample_dir / f"step_{step:02d}.obj"
            png_path = render_dir / f"step_{step:02d}.png"
            if obj_path.exists() and not png_path.exists():
                render_jobs.append((str(obj_path), str(png_path)))

    print(f"Rendering {len(render_jobs)} meshes with {args.workers} workers ...")

    # Render in parallel
    failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(render_one, obj, png, args.resolution): (obj, png)
            for obj, png in render_jobs
        }
        for future in as_completed(futures):
            png_path, success, stderr = future.result()
            if success:
                print(f"  OK: {os.path.basename(os.path.dirname(os.path.dirname(png_path)))}/{os.path.basename(png_path)}")
            else:
                failed += 1
                print(f"  FAIL: {png_path}")
                if stderr:
                    # Print last few lines of stderr for debugging
                    lines = stderr.strip().split("\n")
                    for line in lines[-3:]:
                        print(f"    {line}")

    print(f"\nRendering done. {len(render_jobs) - failed}/{len(render_jobs)} succeeded.")

    # Composite grids
    print("\nCompositing grids ...")
    for sample_dir in sample_dirs:
        if not sample_dir.is_dir():
            continue
        render_dir = sample_dir / "renders"
        if render_dir.exists():
            grid_path = make_grid(
                str(sample_dir), str(render_dir), sample_dir.name,
            )
            if grid_path:
                print(f"  Grid: {grid_path}")

    print("Done.")


if __name__ == "__main__":
    main()
