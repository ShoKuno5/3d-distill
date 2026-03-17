"""Point cloud and mesh normalization.

Canonical normalization:
  1. Translate bbox center to origin
  2. Isotropic scale so that max point norm = 1 (unit sphere)
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class NormTransform:
    """Records the transform applied during normalization."""
    center: np.ndarray  # (3,) translation applied
    scale: float        # scalar scale applied


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
