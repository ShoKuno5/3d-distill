"""Render meshes to multiview images for distributional metrics (FD).

Renders 4 views per mesh at fixed camera positions.
Primary backend: Blender (headless), using _blender_render_mesh.py.
Fallback: pyrender with EGL.

Usage:
    from pipeline.src.evaluation.multiview_renderer import render_multiview

    render_multiview(
        mesh_path="results/predictions/trellis/apple_008/mesh_raw.obj",
        output_dir="results/renders/trellis/apple_008",
        resolution=512,
    )
"""

import os
import subprocess
from pathlib import Path

import numpy as np
import trimesh


# Camera configuration: 4 views at fixed azimuth, 30 deg elevation
AZIMUTHS = [0, 90, 180, 270]
ELEVATION = 30  # degrees
RESOLUTION = 512

# Blender paths (searched in order)
_BLENDER_CANDIDATES = [
    Path(__file__).resolve().parents[3] / "envs" / "blender-3.6.16-linux-x64" / "blender",
    Path("/usr/bin/blender"),
]
_BLENDER_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "_blender_render_mesh.py"

_blender_bin = None


def _find_blender() -> str | None:
    """Find a working Blender binary."""
    global _blender_bin
    if _blender_bin is not None:
        return _blender_bin

    for candidate in _BLENDER_CANDIDATES:
        if candidate.exists():
            _blender_bin = str(candidate)
            return _blender_bin
    return None


def _normalize_mesh_file(mesh_path: str, output_path: str) -> str:
    """Normalize mesh to unit sphere and save to a temp OBJ for Blender."""
    mesh = trimesh.load(mesh_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    centroid = mesh.vertices.mean(axis=0)
    mesh.vertices -= centroid
    max_norm = np.linalg.norm(mesh.vertices, axis=1).max()
    if max_norm > 0:
        mesh.vertices /= max_norm

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mesh.export(output_path)
    return output_path


def render_multiview_blender(
    mesh_path: str,
    output_dir: str,
    resolution: int = RESOLUTION,
    azimuths: list[float] = AZIMUTHS,
    elevation: float = ELEVATION,
) -> list[str]:
    """Render multiview images using Blender headless.

    Uses _blender_render_mesh.py which expects unit-sphere-normalized OBJ meshes.
    """
    blender = _find_blender()
    if blender is None:
        raise RuntimeError("Blender not found")

    # Normalize mesh to temp file
    norm_dir = os.path.join(output_dir, ".tmp")
    norm_path = os.path.join(norm_dir, "normalized.obj")
    _normalize_mesh_file(mesh_path, norm_path)

    output_paths = []
    for az in azimuths:
        out_path = os.path.join(output_dir, f"view_{az}.png")
        if os.path.exists(out_path):
            output_paths.append(out_path)
            continue

        cmd = [
            blender, "--background", "--python", str(_BLENDER_SCRIPT),
            "--", "--input", norm_path,
            "--output", out_path,
            "--resolution", str(resolution),
            "--azimuth", str(float(az)),
            "--elevation", str(float(elevation)),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0 and os.path.exists(out_path):
            output_paths.append(out_path)
        else:
            err = result.stderr[-200:] if result.stderr else "unknown"
            print(f"    Blender render failed for az={az}: {err}")

    # Clean up temp normalized mesh
    try:
        os.remove(norm_path)
        os.rmdir(norm_dir)
    except OSError:
        pass

    return output_paths


def render_multiview(
    mesh_path: str,
    output_dir: str,
    resolution: int = RESOLUTION,
    azimuths: list[float] = AZIMUTHS,
    elevation: float = ELEVATION,
) -> list[str]:
    """Render multiview images from a mesh file.

    Args:
        mesh_path: Path to OBJ/GLB mesh file.
        output_dir: Output directory for rendered images.
        resolution: Render resolution (square).
        azimuths: List of azimuth angles in degrees.
        elevation: Elevation angle in degrees.

    Returns:
        List of output image paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    return render_multiview_blender(mesh_path, output_dir, resolution, azimuths, elevation)


def render_all_meshes(
    predictions_root: str,
    output_dir: str,
    mesh_filename: str = "mesh_raw.obj",
    object_ids: list[str] | None = None,
    resolution: int = RESOLUTION,
) -> int:
    """Render multiview images for all predicted meshes in a directory.

    Args:
        predictions_root: Root directory with {object_id}/{mesh_filename} structure.
        output_dir: Output directory for renders.
        mesh_filename: Mesh filename within each object directory.
        object_ids: If provided, only render these objects.
        resolution: Render resolution.

    Returns:
        Number of successfully rendered objects.
    """
    pred_root = Path(predictions_root)
    success = 0

    if object_ids is None:
        object_ids = sorted(
            d.name for d in pred_root.iterdir()
            if d.is_dir() and (d / mesh_filename).exists()
        )

    for oid in object_ids:
        mesh_path = pred_root / oid / mesh_filename
        if not mesh_path.exists():
            continue

        render_dir = os.path.join(output_dir, oid)

        # Skip if already rendered
        if all(os.path.exists(os.path.join(render_dir, f"view_{az}.png")) for az in AZIMUTHS):
            success += 1
            continue

        try:
            paths = render_multiview(str(mesh_path), render_dir, resolution)
            if len(paths) == len(AZIMUTHS):
                success += 1
            else:
                print(f"  WARNING: {oid}: only {len(paths)}/{len(AZIMUTHS)} views rendered")
        except Exception as e:
            print(f"  ERROR: {oid}: {e}")

    return success
