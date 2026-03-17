"""Similarity ICP alignment with multi-start from 24 cube rotations.

Track A: similarity ICP (rotation + translation + uniform scale)
Track B: center + uniform scale only (no rotation alignment)

Reflection is always forbidden.
Scale is clamped to [0.5, 2.0] to prevent degenerate collapse.

Performance: uses coarse subsampling (5K pts) for the 24-start search,
then refines the best result on the full dense cloud.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


# Number of points used for the coarse ICP search phase
_COARSE_SEARCH_POINTS = 5000
# Scale bounds: since both clouds are unit-sphere normalized, scale should be ~1
_SCALE_MIN = 0.5
_SCALE_MAX = 2.0


@dataclass
class AlignmentResult:
    """Result of alignment procedure."""
    transform_4x4: np.ndarray       # (4, 4) homogeneous transform
    rotation: np.ndarray             # (3, 3) rotation matrix
    translation: np.ndarray          # (3,) translation
    scale: float                     # uniform scale factor
    chamfer_cost: float              # alignment cost (bilateral CD after transform)
    initial_rotation_idx: int        # which of the 24 starts won
    n_icp_iterations: int            # actual iterations used
    method: str                      # 'similarity_icp' or 'scale_only'


def _rotation_matrix(axis: str, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    else:  # z
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def canonical_rotations_24() -> list[np.ndarray]:
    """Generate the 24 proper rotations of a cube (SO(3) octahedral group).

    Enumerates all 6 face orientations (which axis maps to +Z) × 4 in-plane
    rotations around that face.  This guarantees exactly 24 unique rotations,
    covering all axis permutations including cyclic ones (e.g. X→Y→Z→X).
    """
    pi2 = np.pi / 2.0
    # 6 face rotations: which original axis (±) maps to +Z
    face_rotations = [
        np.eye(3),                              # +Z stays +Z
        _rotation_matrix("x", np.pi),           # -Z → +Z
        _rotation_matrix("x", pi2),             # +Y → +Z
        _rotation_matrix("x", -pi2),            # -Y → +Z
        _rotation_matrix("y", -pi2),            # +X → +Z
        _rotation_matrix("y", pi2),             # -X → +Z
    ]
    rots = []
    for R_face in face_rotations:
        for rz in [0, pi2, np.pi, -pi2]:
            R = _rotation_matrix("z", rz) @ R_face
            rots.append(R)
    # De-duplicate (should already be 24 unique, but guard against float noise)
    unique = []
    for R in rots:
        if not any(np.allclose(R, u, atol=1e-6) for u in unique):
            if np.linalg.det(R) > 0:
                unique.append(R)
    return unique


def _bilateral_cd(pts1: np.ndarray, pts2: np.ndarray,
                  tree1: cKDTree = None, tree2: cKDTree = None) -> float:
    """Bilateral Chamfer Distance (L2 squared)."""
    if tree1 is None:
        tree1 = cKDTree(pts1)
    if tree2 is None:
        tree2 = cKDTree(pts2)
    d1, _ = tree2.query(pts1)
    d2, _ = tree1.query(pts2)
    return float((d1**2).mean() + (d2**2).mean())


def _subsample(pts: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """Deterministic subsample of point cloud."""
    if len(pts) <= n:
        return pts
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(pts), n, replace=False)
    return pts[idx]


def _similarity_icp(
    source: np.ndarray,
    target: np.ndarray,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, float, float, int]:
    """Run similarity ICP (rotation + translation + uniform scale).

    Scale is clamped per iteration to prevent degenerate collapse.
    Cost uses one-directional source→target for speed, with final bilateral check.

    Returns:
        (R, t, s, cost, n_iter)
        where transformed = s * (R @ source.T).T + t
    """
    src = source.copy()
    R_acc = np.eye(3)
    t_acc = np.zeros(3)
    s_acc = 1.0
    prev_cost = float("inf")
    n_iter = 0

    target_tree = cKDTree(target)

    for it in range(max_iterations):
        dists, idx = target_tree.query(src)
        matched_target = target[idx]
        R_step, t_step, s_step = _umeyama(src, matched_target)

        # Clamp cumulative scale
        new_s_acc = s_step * s_acc
        if new_s_acc < _SCALE_MIN:
            s_step = _SCALE_MIN / s_acc
        elif new_s_acc > _SCALE_MAX:
            s_step = _SCALE_MAX / s_acc

        src = s_step * (R_step @ src.T).T + t_step

        R_acc = R_step @ R_acc
        t_acc = s_step * (R_step @ t_acc) + t_step
        s_acc = s_step * s_acc

        # One-directional cost for speed (src→target)
        dists_new, _ = target_tree.query(src)
        cost = float((dists_new**2).mean())
        n_iter = it + 1

        if abs(prev_cost - cost) < tolerance:
            break
        prev_cost = cost

    # Final bilateral CD for the cost metric
    final_cost = _bilateral_cd(src, target, tree2=target_tree)
    return R_acc, t_acc, s_acc, final_cost, n_iter


def _umeyama(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Umeyama alignment: (R, t, s) minimizing ||target - (s*R*source + t)||^2."""
    mu_s = source.mean(axis=0)
    mu_t = target.mean(axis=0)
    src_c = source - mu_s
    tgt_c = target - mu_t

    var_s = (src_c**2).sum() / len(source)

    cov = (tgt_c.T @ src_c) / len(source)
    U, D, Vt = np.linalg.svd(cov)

    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1

    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / var_s if var_s > 1e-12 else 1.0

    # Clamp per-step scale
    s = max(_SCALE_MIN, min(_SCALE_MAX, s))

    t = mu_t - s * (R @ mu_s)

    return R, t, s


def align_similarity_icp(
    pred_pts: np.ndarray,
    gt_pts: np.ndarray,
    max_iterations: int = 100,
    n_initial_rotations: int = 24,
) -> AlignmentResult:
    """Multi-start similarity ICP alignment (Track A).

    Two-phase approach for performance:
    1. Coarse search: subsample to ~5K pts, try all 24 rotations
    2. Refinement: run ICP on the full dense cloud from the best start

    Args:
        pred_pts: (N, 3) predicted point cloud (pre-normalized to unit sphere).
        gt_pts: (M, 3) GT point cloud (in canonical frame).
        max_iterations: Max ICP iterations per start.
        n_initial_rotations: Number of initial rotations (24 for cube group).

    Returns:
        AlignmentResult with best transform.
    """
    rotations = canonical_rotations_24()[:n_initial_rotations]

    # Phase 1: Coarse search with subsampled points
    pred_coarse = _subsample(pred_pts, _COARSE_SEARCH_POINTS, seed=0)
    gt_coarse = _subsample(gt_pts, _COARSE_SEARCH_POINTS, seed=0)

    best_coarse_cost = float("inf")
    best_coarse_idx = 0
    best_coarse_R = np.eye(3)
    best_coarse_t = np.zeros(3)
    best_coarse_s = 1.0

    for idx, R_init in enumerate(rotations):
        rotated = (R_init @ pred_coarse.T).T
        R_icp, t_icp, s_icp, cost, _ = _similarity_icp(
            rotated, gt_coarse, max_iterations=50
        )
        R_total = R_icp @ R_init
        if np.linalg.det(R_total) < 0:
            continue
        if cost < best_coarse_cost:
            best_coarse_cost = cost
            best_coarse_idx = idx
            best_coarse_R = R_total
            best_coarse_t = t_icp
            best_coarse_s = s_icp

    # Phase 2: Refine on full dense cloud from the best coarse start
    pred_init = best_coarse_s * (best_coarse_R @ pred_pts.T).T + best_coarse_t
    R_refine, t_refine, s_refine, cost_refine, n_iter = _similarity_icp(
        pred_init, gt_pts, max_iterations=max_iterations
    )

    # Compose: refine ∘ coarse
    R_final = R_refine @ best_coarse_R
    s_final = s_refine * best_coarse_s
    t_final = s_refine * (R_refine @ best_coarse_t) + t_refine

    # Clamp composed scale to prevent degenerate collapse across phases
    s_final = max(_SCALE_MIN, min(_SCALE_MAX, s_final))

    if np.linalg.det(R_final) < 0:
        R_final = best_coarse_R
        s_final = best_coarse_s
        t_final = best_coarse_t
        cost_refine = best_coarse_cost

    T = np.eye(4)
    T[:3, :3] = s_final * R_final
    T[:3, 3] = t_final

    return AlignmentResult(
        transform_4x4=T,
        rotation=R_final,
        translation=t_final,
        scale=s_final,
        chamfer_cost=cost_refine,
        initial_rotation_idx=best_coarse_idx,
        n_icp_iterations=n_iter,
        method="similarity_icp",
    )


def align_scale_only(
    pred_pts: np.ndarray,
    gt_pts: np.ndarray,
) -> AlignmentResult:
    """Track B alignment: center + uniform scale only, no rotation."""
    T = np.eye(4)
    cost = _bilateral_cd(pred_pts, gt_pts)
    return AlignmentResult(
        transform_4x4=T,
        rotation=np.eye(3),
        translation=np.zeros(3),
        scale=1.0,
        chamfer_cost=cost,
        initial_rotation_idx=-1,
        n_icp_iterations=0,
        method="scale_only",
    )


def apply_alignment(pts: np.ndarray, result: AlignmentResult) -> np.ndarray:
    """Apply alignment transform to points."""
    T = result.transform_4x4
    R_scaled = T[:3, :3]
    t = T[:3, 3]
    return (R_scaled @ pts.T).T + t
