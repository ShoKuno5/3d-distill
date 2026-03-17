"""Surface sampling from meshes — area-proportional."""

import numpy as np
import trimesh


def sample_surface(mesh: trimesh.Trimesh, n_points: int, seed: int = 0) -> np.ndarray:
    """Sample points from mesh surface with area-proportional weighting.

    Args:
        mesh: Input trimesh.
        n_points: Number of points to sample.
        seed: Random seed for reproducibility.

    Returns:
        (N, 3) point cloud as float64.
    """
    # trimesh uses numpy's global random state; set seed for reproducibility
    rng_state = np.random.get_state()
    np.random.seed(seed)
    try:
        pts, _ = trimesh.sample.sample_surface(mesh, n_points)
    finally:
        np.random.set_state(rng_state)
    return np.array(pts, dtype=np.float64)
