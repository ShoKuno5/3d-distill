#!/usr/bin/env python3
"""Render all model outputs side-by-side for visual comparison.

Produces a grid image: rows = objects, cols = [input image, model1, model2, ...].
Uses matplotlib for rendering (no OpenGL dependency).

Usage:
    cd /mnt/workspace/kuno/distillation
    envs/hunyuan3d-venv/bin/python pipeline/scripts/render_comparison.py \
        --config experiments/toys4k_baseline/config.yaml \
        --output results/toys4k/comparison.png
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.utils.inference_config import get_model_config, load_and_filter_samples


# Lazy-init pyrender to avoid import overhead when not needed
_renderer = None


def _get_renderer(resolution: int):
    global _renderer
    if _renderer is not None and _renderer.viewport_width == resolution:
        return _renderer
    import pyrender
    _renderer = pyrender.OffscreenRenderer(resolution, resolution)
    return _renderer


def render_mesh_pyrender(mesh_path: str, resolution: int = 384,
                         azimuth: float = 150.0, elevation: float = 20.0) -> np.ndarray:
    """Render a mesh using pyrender (proper rasterization). Returns RGB array."""
    import trimesh
    import pyrender

    mesh = trimesh.load(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    # Center and normalize to [-0.5, 0.5]
    bounds_min, bounds_max = mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)
    center = (bounds_min + bounds_max) / 2.0
    scale = max(bounds_max - bounds_min)
    if scale > 0:
        mesh.vertices = (mesh.vertices - center) / scale

    # Set uniform color
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh,
        face_colors=np.full((len(mesh.faces), 4), [160, 160, 200, 255], dtype=np.uint8),
    )

    # Build pyrender scene
    scene = pyrender.Scene(bg_color=[255, 255, 255, 255], ambient_light=[0.3, 0.3, 0.3])
    py_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    scene.add(py_mesh)

    # Camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 6.0)
    az_rad = np.radians(azimuth)
    el_rad = np.radians(elevation)
    dist = 2.5
    cx = dist * np.cos(el_rad) * np.sin(az_rad)
    cy = dist * np.sin(el_rad)
    cz = dist * np.cos(el_rad) * np.cos(az_rad)
    cam_pos = np.array([cx, cy, cz])

    # Look-at matrix
    forward = -cam_pos / np.linalg.norm(cam_pos)
    right = np.cross(forward, np.array([0, 1, 0]))
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        right = np.array([1, 0, 0])
    else:
        right = right / right_norm
    up = np.cross(right, forward)

    cam_pose = np.eye(4)
    cam_pose[:3, 0] = right
    cam_pose[:3, 1] = up
    cam_pose[:3, 2] = -forward
    cam_pose[:3, 3] = cam_pos
    scene.add(camera, pose=cam_pose)

    # Directional light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=cam_pose)

    r = _get_renderer(resolution)
    color, _ = r.render(scene)
    return color


def load_input_image(path: str, resolution: int) -> np.ndarray:
    """Load and resize input image to match render resolution."""
    img = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    composite = Image.alpha_composite(bg, img).convert("RGB")
    composite = composite.resize((resolution, resolution), Image.LANCZOS)
    return np.array(composite)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None, help="Output image path")
    parser.add_argument("--resolution", type=int, default=384, help="Per-cell resolution")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    samples = load_and_filter_samples(cfg, max_samples_override=args.max_samples)
    model_names = [m["name"] for m in cfg["models"]]

    if args.output is None:
        args.output = os.path.join(cfg["output_root"], "comparison.png")

    res = args.resolution
    n_rows = len(samples)
    n_cols = 1 + len(model_names)

    print(f"Rendering {n_rows} samples x {len(model_names)} models @ {res}px")

    grid = np.full((n_rows * res, n_cols * res, 3), 240, dtype=np.uint8)

    # Input images (column 0)
    for row, s in enumerate(samples):
        try:
            grid[row * res:(row + 1) * res, 0:res] = load_input_image(s.input_image, res)
        except Exception as e:
            print(f"  Input error {s.object_id}: {e}")

    # Render meshes
    total = 0
    for row, s in enumerate(samples):
        for col_off, mname in enumerate(model_names):
            col = col_off + 1
            mcfg = get_model_config(cfg, mname)
            mesh_path = os.path.join(mcfg["predictions_root"], s.object_id, mcfg["mesh_filename"])
            total += 1
            if not os.path.exists(mesh_path):
                print(f"  MISSING: {mname}/{s.object_id}")
                continue
            try:
                img = render_mesh_pyrender(mesh_path, res)
                grid[row * res:(row + 1) * res, col * res:(col + 1) * res] = img
            except Exception as e:
                print(f"  FAILED {mname}/{s.object_id}: {e}")
            if total % 10 == 0:
                print(f"  Progress: {total}/{n_rows * len(model_names)}")

    print(f"  Done: {total}/{n_rows * len(model_names)}")

    # Add labels
    result = Image.fromarray(grid)
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(result)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
        small_font = font

    headers = ["Input"] + model_names
    for col, header in enumerate(headers):
        x = col * res + res // 2
        draw.text((x, 2), header, fill=(0, 0, 0), font=font, anchor="mt")

    for row, s in enumerate(samples):
        y = row * res + res - 4
        draw.text((4, y), s.object_id, fill=(80, 80, 80), font=small_font, anchor="lb")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    result.save(args.output)
    print(f"\nSaved: {args.output} ({result.size[0]}x{result.size[1]}px)")


if __name__ == "__main__":
    main()
