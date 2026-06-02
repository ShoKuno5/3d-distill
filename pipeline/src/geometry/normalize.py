"""Point cloud and mesh normalization.

Two conventions:
  - unit_sphere: center bbox, scale so max point norm = 1.
  - longest_axis: center bbox, scale so the LONGEST bbox axis has extent 2.0
    (range [-1, 1] on that axis). This matches the Toys4k GT point clouds
    (pc10K.npz), which are per-object longest-axis normalized and centered.
    Use this so predictions live in the same frame as GT and the meaningful
    quantity is RELATIVE-extent / aspect-ratio fidelity (absolute scale is not
    recoverable from the GT).
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class NormTransform:
    """Records the transform applied during normalization."""
    center: np.ndarray  # (3,) translation applied
    scale: float        # scalar scale applied (divisor)


def normalize(pts: np.ndarray, method: str = "unit_sphere") -> tuple[np.ndarray, NormTransform]:
    """Dispatch normalization by method name ('unit_sphere' | 'longest_axis')."""
    if method == "unit_sphere":
        return normalize_to_unit_sphere(pts)
    if method == "longest_axis":
        return normalize_longest_axis(pts)
    raise ValueError(f"Unknown normalization method: {method!r}")


def normalize_longest_axis(pts: np.ndarray) -> tuple[np.ndarray, NormTransform]:
    """Center at bbox center, isotropic scale so the longest bbox axis -> extent 2.0.

    Matches the Toys4k GT convention. The divisor is half the longest extent,
    so after normalization the longest axis spans [-1, 1].

    Args:
        pts: (N, 3) point cloud.

    Returns:
        (normalized_pts, transform)
    """
    bbox_min = pts.min(axis=0)
    bbox_max = pts.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    centered = pts - center
    half_extent = (bbox_max - bbox_min).max() / 2.0
    if half_extent > 0:
        normalized = centered / half_extent
    else:
        normalized = centered
        half_extent = 1.0
    return normalized, NormTransform(center=center, scale=half_extent)


def normalize_to_unit_sphere(pts: np.ndarray) -> tuple[np.ndarray, NormTransform]:
    """Center at bbox center, scale so max point norm = 1.

    Args:
        pts: (N, 3) point cloud.

    Returns:
        (normalized_pts, transform)
    """
    bbox_min = pts.min(axis=0)
    bbox_max = pts.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    centered = pts - center
    max_norm = np.linalg.norm(centered, axis=1).max()
    if max_norm > 0:
        normalized = centered / max_norm
    else:
        normalized = centered
        max_norm = 1.0
    return normalized, NormTransform(center=center, scale=max_norm)
