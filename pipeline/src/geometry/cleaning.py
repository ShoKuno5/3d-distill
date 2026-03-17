"""Minimal mesh cleaning — same rules for all models.

Allowed operations only:
  - NaN/Inf vertex removal
  - Degenerate face removal
  - Unreferenced vertex removal
  - Tiny disconnected component removal (configurable threshold)
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import trimesh


@dataclass
class CleaningReport:
    """Record of what cleaning did."""
    original_vertices: int
    original_faces: int
    cleaned_vertices: int
    cleaned_faces: int
    nan_vertices_removed: int
    inf_vertices_removed: int
    degenerate_faces_removed: int
    unreferenced_vertices_removed: int
    tiny_components_removed: int
    is_empty: bool
    is_broken: bool
    error: Optional[str] = None


def clean_mesh(
    mesh: trimesh.Trimesh,
    remove_nan: bool = True,
    remove_inf: bool = True,
    remove_degenerate: bool = True,
    remove_unreferenced: bool = True,
    remove_tiny_components: bool = True,
    tiny_threshold: float = 0.01,
) -> tuple[trimesh.Trimesh, CleaningReport]:
    """Apply minimal cleaning to a prediction mesh.

    Args:
        mesh: Input trimesh.
        remove_nan: Remove vertices with NaN coords.
        remove_inf: Remove vertices with Inf coords.
        remove_degenerate: Remove zero-area faces.
        remove_unreferenced: Remove vertices not in any face.
        remove_tiny_components: Remove components with < threshold fraction of faces.
        tiny_threshold: Fraction of total faces below which a component is removed.

    Returns:
        (cleaned_mesh, report)
    """
    orig_v = len(mesh.vertices)
    orig_f = len(mesh.faces)
    nan_removed = 0
    inf_removed = 0
    degen_removed = 0
    unref_removed = 0
    tiny_removed = 0

    vertices = np.array(mesh.vertices, dtype=np.float64)
    faces = np.array(mesh.faces, dtype=np.int64)

    # Remove NaN vertices
    if remove_nan:
        valid = ~np.any(np.isnan(vertices), axis=1)
        nan_removed = int((~valid).sum())
        if nan_removed > 0:
            vertices, faces = _remove_vertices(vertices, faces, ~valid)

    # Remove Inf vertices
    if remove_inf:
        valid = ~np.any(np.isinf(vertices), axis=1)
        inf_removed = int((~valid).sum())
        if inf_removed > 0:
            vertices, faces = _remove_vertices(vertices, faces, ~valid)

    # Remove degenerate faces (zero area)
    if remove_degenerate and len(faces) > 0:
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        areas = np.linalg.norm(cross, axis=1)
        valid_faces = areas > 1e-10
        degen_removed = int((~valid_faces).sum())
        faces = faces[valid_faces]

    # Remove unreferenced vertices
    if remove_unreferenced and len(faces) > 0:
        used = np.unique(faces)
        n_before = len(vertices)
        remap = np.full(len(vertices), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        vertices = vertices[used]
        faces = remap[faces]
        unref_removed = n_before - len(vertices)

    # Remove tiny disconnected components
    if remove_tiny_components and len(faces) > 0 and tiny_threshold > 0:
        cleaned = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        components = cleaned.split(only_watertight=False)
        if len(components) > 1:
            total_faces = sum(len(c.faces) for c in components)
            threshold_count = int(total_faces * tiny_threshold)
            kept = [c for c in components if len(c.faces) >= threshold_count]
            removed_count = len(components) - len(kept)
            if kept and removed_count > 0:
                tiny_removed = removed_count
                merged = trimesh.util.concatenate(kept)
                vertices = np.array(merged.vertices, dtype=np.float64)
                faces = np.array(merged.faces, dtype=np.int64)

    is_empty = len(faces) == 0 or len(vertices) == 0
    is_broken = is_empty

    cleaned_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    report = CleaningReport(
        original_vertices=orig_v,
        original_faces=orig_f,
        cleaned_vertices=len(vertices),
        cleaned_faces=len(faces),
        nan_vertices_removed=nan_removed,
        inf_vertices_removed=inf_removed,
        degenerate_faces_removed=degen_removed,
        unreferenced_vertices_removed=unref_removed,
        tiny_components_removed=tiny_removed,
        is_empty=is_empty,
        is_broken=is_broken,
    )
    return cleaned_mesh, report


def _remove_vertices(
    vertices: np.ndarray, faces: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Remove vertices indicated by mask and update face indices."""
    keep = ~mask
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[keep] = np.arange(keep.sum())
    vertices = vertices[keep]
    # Remove faces referencing removed vertices
    valid_faces = np.all(keep[faces], axis=1)
    faces = remap[faces[valid_faces]]
    return vertices, faces
