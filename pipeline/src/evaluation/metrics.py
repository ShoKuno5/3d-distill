"""Geometry evaluation metrics.

All metrics operate on point clouds (not raw meshes).
CD definition: bilateral mean of squared L2 distances.
F-score: harmonic mean of precision and recall at distance threshold.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class GeometryMetrics:
    """Per-sample geometry evaluation results."""
    chamfer_distance: float         # bilateral mean of squared L2
    chamfer_pred_to_gt: float       # one-directional
    chamfer_gt_to_pred: float       # one-directional
    hausdorff: float                # max of directed hausdorff both ways
    f_score_001: float              # F-score @ tau=0.01
    f_score_002: float              # F-score @ tau=0.02


def compute_geometry_metrics(
    pred_pts: np.ndarray,
    gt_pts: np.ndarray,
    f_thresholds: tuple[float, ...] = (0.01, 0.02),
) -> GeometryMetrics:
    """Compute all geometry metrics between two point clouds.

    Both point clouds should be in the same coordinate frame
    (post-alignment, post-normalization).

    Args:
        pred_pts: (N, 3) predicted points.
        gt_pts: (M, 3) GT points.
        f_thresholds: Thresholds for F-score computation.

    Returns:
        GeometryMetrics dataclass.
    """
    tree_pred = cKDTree(pred_pts)
    tree_gt = cKDTree(gt_pts)

    # Distances: pred->gt and gt->pred
    d_pred_to_gt, _ = tree_gt.query(pred_pts)
    d_gt_to_pred, _ = tree_pred.query(gt_pts)

    # Chamfer Distance (bilateral, L2 squared)
    cd_p2g = float((d_pred_to_gt**2).mean())
    cd_g2p = float((d_gt_to_pred**2).mean())
    cd = cd_p2g + cd_g2p

    # Hausdorff
    hausdorff = float(max(d_pred_to_gt.max(), d_gt_to_pred.max()))

    # F-scores
    f_scores = {}
    for tau in f_thresholds:
        precision = float((d_pred_to_gt < tau).mean())
        recall = float((d_gt_to_pred < tau).mean())
        if precision + recall > 0:
            f_scores[tau] = 2 * precision * recall / (precision + recall)
        else:
            f_scores[tau] = 0.0

    return GeometryMetrics(
        chamfer_distance=cd,
        chamfer_pred_to_gt=cd_p2g,
        chamfer_gt_to_pred=cd_g2p,
        hausdorff=hausdorff,
        f_score_001=f_scores.get(0.01, 0.0),
        f_score_002=f_scores.get(0.02, 0.0),
    )


@dataclass
class ExtentMetrics:
    """Relative-extent / aspect-ratio fidelity (the scale-honest signal).

    Computed on the (longest-axis-normalized) clouds BEFORE alignment, on
    bbox extents sorted descending so they are rotation/axis-permutation
    invariant. Because both clouds are longest-axis normalized, ext_ratio_long
    is ~1 by construction; the informative quantity is the SHORT axis — does a
    flat coin/blade keep its thin axis, or does 1-step generation inflate it.
    """
    ext_ratio_long: float    # pred/gt longest-axis extent  (~1 under longest-axis norm)
    ext_ratio_mid: float     # pred/gt middle-axis extent
    ext_ratio_short: float   # pred/gt shortest-axis extent
    short_axis_ratio: float  # alias of ext_ratio_short (the headline collapse signal)
    bbox_diag_ratio: float   # ||pred bbox diag|| / ||gt bbox diag||
    aspect_error: float      # |log( (short/long)_pred / (short/long)_gt )|
    log_scale_error: float   # |log(bbox_diag_ratio)|


def _sorted_extents(pts: np.ndarray) -> np.ndarray:
    """bbox extents per axis, sorted descending: [long, mid, short]."""
    ext = pts.max(axis=0) - pts.min(axis=0)
    return np.sort(ext)[::-1]


def compute_extent_metrics(pred_pts: np.ndarray, gt_pts: np.ndarray) -> ExtentMetrics:
    """Relative-extent / aspect-ratio fidelity. Alignment-free (uses sorted extents)."""
    eps = 1e-8
    ep = _sorted_extents(pred_pts)
    eg = _sorted_extents(gt_pts)
    ratios = ep / np.maximum(eg, eps)
    diag_p = float(np.linalg.norm(ep))
    diag_g = float(np.linalg.norm(eg))
    bbox_diag_ratio = diag_p / max(diag_g, eps)
    aspect_p = ep[2] / max(ep[0], eps)
    aspect_g = eg[2] / max(eg[0], eps)
    aspect_error = abs(float(np.log(max(aspect_p, eps) / max(aspect_g, eps))))
    log_scale_error = abs(float(np.log(max(bbox_diag_ratio, eps))))
    return ExtentMetrics(
        ext_ratio_long=float(ratios[0]),
        ext_ratio_mid=float(ratios[1]),
        ext_ratio_short=float(ratios[2]),
        short_axis_ratio=float(ratios[2]),
        bbox_diag_ratio=bbox_diag_ratio,
        aspect_error=aspect_error,
        log_scale_error=log_scale_error,
    )


def is_extent_collapse(
    em: ExtentMetrics,
    short_lo: float = 0.5,
    short_hi: float = 2.0,
    aspect_max: float = float(np.log(2.0)),
) -> bool:
    """Flag a degenerate extent collapse (a non-empty but wrongly-proportioned shape).

    Faithful reconstructions keep each axis within ~2x of GT; empirical 1-step
    collapses sit at 3-16x on the short axis. A factor-of-2 band separates them.
    """
    return (
        em.short_axis_ratio < short_lo
        or em.short_axis_ratio > short_hi
        or em.aspect_error > aspect_max
    )


@dataclass
class MeshQualityStats:
    """Mesh quality statistics for a single prediction."""
    vertex_count: int
    face_count: int
    connected_components: int
    is_watertight: bool
    is_manifold: bool     # edge-manifold (each edge shared by ≤ 2 faces)
    has_self_intersections: Optional[bool]  # None if check skipped
    is_empty: bool
    is_broken: bool


def compute_mesh_quality(mesh) -> MeshQualityStats:
    """Compute mesh quality statistics.

    Args:
        mesh: trimesh.Trimesh object.

    Returns:
        MeshQualityStats dataclass.
    """
    import trimesh

    n_verts = len(mesh.vertices)
    n_faces = len(mesh.faces)
    is_empty = n_verts == 0 or n_faces == 0

    if is_empty:
        return MeshQualityStats(
            vertex_count=n_verts,
            face_count=n_faces,
            connected_components=0,
            is_watertight=False,
            is_manifold=False,
            has_self_intersections=None,
            is_empty=True,
            is_broken=True,
        )

    # Connected components
    try:
        components = mesh.split(only_watertight=False)
        n_components = len(components)
    except Exception:
        n_components = -1

    # Watertight
    try:
        watertight = bool(mesh.is_watertight)
    except Exception:
        watertight = False

    # Edge-manifold (no edge shared by >2 faces, no isolated edges)
    try:
        # trimesh doesn't directly expose is_manifold; approximate via euler check
        # A closed manifold satisfies V - E + F = 2 * (components - genus)
        # For our purposes, check if the mesh has consistent face winding
        edges = mesh.edges_sorted
        from collections import Counter
        edge_counts = Counter(map(tuple, edges))
        max_sharing = max(edge_counts.values()) if edge_counts else 0
        is_manifold = max_sharing <= 2
    except Exception:
        is_manifold = False

    # Self-intersections: expensive, skip by default but flag
    has_self_intersections = None  # not checked

    is_broken = is_empty or n_faces < 4

    return MeshQualityStats(
        vertex_count=n_verts,
        face_count=n_faces,
        connected_components=n_components,
        is_watertight=watertight,
        is_manifold=is_manifold,
        has_self_intersections=has_self_intersections,
        is_empty=is_empty,
        is_broken=is_broken,
    )


def bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval.

    Returns:
        (mean, ci_low, ci_high)
    """
    rng = np.random.RandomState(seed)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")

    means = np.array([
        rng.choice(values, size=n, replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    alpha = 1 - confidence
    ci_low = float(np.percentile(means, 100 * alpha / 2))
    ci_high = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return float(values.mean()), ci_low, ci_high
