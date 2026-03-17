#!/usr/bin/env python3
"""Render normal-map comparison grid for all models.

Uses TRELLIS MeshRenderer (nvdiffrast GPU rasterizer) for high-quality
normal rendering, then composes a summary grid.

Must run from trellis env:
    cd models/trellis && PYTHONPATH=. SPCONV_ALGO=native \
    ../../envs/miniconda3/envs/trellis/bin/python \
    ../../pipeline/scripts/render_normals_grid.py \
    --config ../../experiments/examples_qual/config.yaml
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

os.environ.setdefault("SPCONV_ALGO", "native")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from trellis.representations import MeshExtractResult
from trellis.renderers import MeshRenderer
from trellis.utils.render_utils import (
    render_frames,
    yaw_pitch_r_fov_to_extrinsics_intrinsics,
)
from src.utils.inference_config import get_model_config, load_and_filter_samples

# Render settings
RESOLUTION = 512
AZIMUTHS = [i * np.pi / 4 for i in range(8)]
ELEVATION = 20 * np.pi / 180
RADIUS = 2.0
FOV = 40
BG_COLOR = (1, 1, 1)

# Grid settings
VIEW_INDICES = [0, 2, 4, 7]  # front(0°), right(90°), back(180°), 3/4(315°)
VIEW_LABELS = ["Front", "Right", "Back", "3/4 Rear"]
CELL_SIZE = 256


def get_cameras():
    yaws = AZIMUTHS
    pitchs = [ELEVATION] * 8
    return yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, RADIUS, FOV)


MAX_FACES = 500_000  # decimate if more faces than this

# Coordinate system: TRELLIS renderer expects Z-up.
# Hunyuan3D models output Y-up, so we rotate Y-up → Z-up.
_YUP_TO_ZUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)

# Model-specific up-axis (empirically verified)
MODEL_UP_AXIS = {
    "trellis": "z",
    "trellis2": "z",
    "hunyuan3d": "y",
    "hunyuan3d21": "y",
}


def load_obj_as_mesh_result(obj_path: str, model_name: str = None) -> MeshExtractResult:
    """Load an OBJ mesh and wrap as MeshExtractResult for TRELLIS renderer."""
    import trimesh
    mesh = trimesh.load(obj_path, process=False, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    # Coordinate alignment: rotate Y-up models to Z-up (renderer convention)
    if model_name and MODEL_UP_AXIS.get(model_name) == "y":
        mesh.vertices = mesh.vertices @ _YUP_TO_ZUP

    # Decimate if too many faces (proper quadric decimation)
    if len(mesh.faces) > MAX_FACES:
        mesh = mesh.simplify_quadric_decimation(face_count=MAX_FACES)
        print(f" [decimated→{len(mesh.faces)}f]", end="", flush=True)

    vertices = torch.tensor(mesh.vertices, dtype=torch.float32).cuda()
    faces = torch.tensor(mesh.faces, dtype=torch.long).cuda()

    # Normalize to [-0.5, 0.5]
    center = (vertices.max(0).values + vertices.min(0).values) / 2
    scale = (vertices.max(0).values - vertices.min(0).values).max()
    if scale > 0:
        vertices = (vertices - center) / scale

    return MeshExtractResult(vertices=vertices, faces=faces, vertex_attrs=None)


def render_normals(mesh_result, extrinsics, intrinsics) -> list[np.ndarray]:
    """Render normal maps for all 8 viewpoints. Returns list of RGB arrays."""
    options = {"resolution": RESOLUTION, "bg_color": BG_COLOR}
    result = render_frames(mesh_result, extrinsics, intrinsics, options, verbose=False)
    return result["normal"]


def compose_grid(
    objects: list[str],
    input_images: dict[str, str],
    model_names: list[str],
    renders: dict[tuple[str, str], list[np.ndarray]],
    output_path: str,
):
    """Compose summary grid: rows = objects, sub-rows = models, cols = views."""
    n_views = len(VIEW_INDICES)
    n_models = len(model_names)
    label_w = 140

    # Each object block: 1 row input + n_models rows
    rows_per_obj = 1 + n_models
    total_rows = len(objects) * rows_per_obj
    header_h = 28
    obj_separator = 4

    img_w = label_w + n_views * (CELL_SIZE + 2) + 2
    img_h = header_h + len(objects) * (rows_per_obj * (CELL_SIZE + 2) + obj_separator)

    grid = Image.new("RGB", (img_w, img_h), (255, 255, 255))
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
        small = font

    # Column headers
    for j, vlabel in enumerate(VIEW_LABELS):
        x = label_w + 2 + j * (CELL_SIZE + 2) + CELL_SIZE // 2
        draw.text((x, 6), vlabel, fill=(80, 80, 80), font=font, anchor="mt")

    y = header_h
    for obj_id in objects:
        # Input row
        draw.text((4, y + CELL_SIZE // 2 - 6), obj_id, fill=(0, 0, 0), font=font)
        inp_path = input_images.get(obj_id)
        if inp_path and os.path.exists(inp_path):
            inp = Image.open(inp_path).convert("RGBA")
            bg = Image.new("RGBA", inp.size, (255, 255, 255, 255))
            inp = Image.alpha_composite(bg, inp).convert("RGB")
            inp = inp.resize((CELL_SIZE, CELL_SIZE), Image.LANCZOS)
            grid.paste(inp, (label_w + 2, y))
        y += CELL_SIZE + 2

        # Model rows
        for mname in model_names:
            draw.text((8, y + CELL_SIZE // 2 - 6), mname, fill=(60, 60, 60), font=small)
            frames = renders.get((obj_id, mname))
            if frames is not None:
                for j, vi in enumerate(VIEW_INDICES):
                    x = label_w + 2 + j * (CELL_SIZE + 2)
                    frame = Image.fromarray(frames[vi]).resize(
                        (CELL_SIZE, CELL_SIZE), Image.LANCZOS
                    )
                    grid.paste(frame, (x, y))
            else:
                for j in range(n_views):
                    x = label_w + 2 + j * (CELL_SIZE + 2)
                    placeholder = Image.new("RGB", (CELL_SIZE, CELL_SIZE), (220, 220, 220))
                    d = ImageDraw.Draw(placeholder)
                    d.text((CELL_SIZE // 2, CELL_SIZE // 2), "missing", fill=(160, 160, 160), font=small, anchor="mm")
                    grid.paste(placeholder, (x, y))
            y += CELL_SIZE + 2

        y += obj_separator

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    grid.save(output_path)
    print(f"Saved: {output_path} ({grid.size[0]}x{grid.size[1]})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    samples = load_and_filter_samples(cfg, max_samples_override=args.max_samples)
    model_cfgs = cfg["models"]
    model_names = [m["name"] for m in model_cfgs]
    output_root = cfg["output_root"]

    if args.output is None:
        args.output = os.path.join(output_root, "summary_normals.png")

    print(f"Rendering {len(samples)} objects x {len(model_names)} models")
    extrinsics, intrinsics = get_cameras()

    # Collect input image paths
    input_images = {s.object_id: s.input_image for s in samples}

    # Render all meshes
    renders = {}
    for mcfg in model_cfgs:
        mname = mcfg["name"]
        pred_root = mcfg["predictions_root"]
        mesh_fn = mcfg["mesh_filename"]

        for s in samples:
            mesh_path = os.path.join(pred_root, s.object_id, mesh_fn)
            if not os.path.exists(mesh_path):
                print(f"  MISSING: {mname}/{s.object_id}")
                continue

            print(f"  Rendering {mname}/{s.object_id}...", end="", flush=True)
            try:
                mesh_result = load_obj_as_mesh_result(mesh_path, model_name=mname)
                frames = render_normals(mesh_result, extrinsics, intrinsics)
                renders[(s.object_id, mname)] = frames

                # Also save individual PNGs for later reuse
                render_dir = os.path.join(pred_root, s.object_id, "renders")
                os.makedirs(render_dir, exist_ok=True)
                az_labels = ["000", "045", "090", "135", "180", "225", "270", "315"]
                for i, frame in enumerate(frames):
                    Image.fromarray(frame).save(os.path.join(render_dir, f"normal_{az_labels[i]}.png"))

                print(f" OK ({len(mesh_result.vertices)} verts)")
            except Exception as e:
                print(f" FAILED: {e}")

    # Compose grid
    objects = [s.object_id for s in samples]
    compose_grid(objects, input_images, model_names, renders, args.output)


if __name__ == "__main__":
    main()
