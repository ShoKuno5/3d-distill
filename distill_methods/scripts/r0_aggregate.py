#!/usr/bin/env python3
"""R0 manual CD aggregator from saved per-sample alignments.

Reads:
    normalized_predictions/<model>/<oid>/eval_pts.npz       (sampled pred, normalized, pre-alignment)
    normalized_predictions/<model>/<oid>/alignment_track_a.json   (transform)

Loads GT from Toys4k 105 manifest, normalizes/samples same as run_eval.py,
applies the saved alignment, computes Chamfer Distance (L2_squared), reports
per-model mean/median.

Use this when run_eval.py's ProcessPoolExecutor hangs at shutdown after
worker crashes — the on-disk per-sample data is still valid.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import trimesh
import yaml

PROJECT_DIR = Path("/inspire/qb-ilm/project/qproject-assement/zhangkaipeng-24043/sk5/repos/3d-gen-eval")
sys.path.insert(0, str(PROJECT_DIR / "pipeline"))

from src.geometry.normalize import normalize_to_unit_sphere  # noqa: E402
from src.geometry.sampling import sample_surface             # noqa: E402
from src.geometry.alignment import apply_alignment, AlignmentResult  # noqa: E402
from src.evaluation.metrics import compute_geometry_metrics   # noqa: E402
from src.data.toys4k import load_gt_pointcloud               # noqa: E402


def load_eval_pts(path):
    d = np.load(path)
    return d["points"].astype(np.float64)


def load_alignment(path):
    with open(path) as f:
        j = json.load(f)
    return AlignmentResult(
        transform_4x4=np.asarray(j["transform_4x4"], dtype=np.float64),
        rotation=np.asarray(j["rotation"], dtype=np.float64),
        translation=np.asarray(j["translation"], dtype=np.float64),
        scale=float(j["scale"]),
        chamfer_cost=float(j["chamfer_cost"]),
        initial_rotation_idx=int(j["initial_rotation_idx"]),
        n_icp_iterations=int(j["n_icp_iterations"]),
        method=str(j["method"]),
    )


def load_gt_pts(pc_path: str, n_eval: int = 16384, rng: np.random.Generator | None = None):
    """Load Toys4k pc10K.npz via repo helper, normalize to unit sphere, sample n_eval points."""
    pts, _normals = load_gt_pointcloud(pc_path)
    pts = pts.astype(np.float64)
    pts_norm, _ = normalize_to_unit_sphere(pts)
    if rng is None:
        rng = np.random.default_rng(42)
    if len(pts_norm) >= n_eval:
        idx = rng.choice(len(pts_norm), n_eval, replace=False)
    else:
        idx = rng.choice(len(pts_norm), n_eval, replace=True)
    return pts_norm[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--per-sample-csv", default=None)
    ap.add_argument("--n-eval", type=int, default=16384)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out_root = Path(cfg["output_root"])
    norm_root = out_root / "normalized_predictions"

    test_manifest = cfg["dataset"]["test_manifest"]
    rows = list(csv.DictReader(open(test_manifest)))
    print(f"Manifest: {len(rows)} samples", file=sys.stderr)
    rows_by_oid = {r["object_id"]: r for r in rows}

    f_thresholds = cfg["metrics"]["f_score"]["thresholds"]

    # Pre-load and cache GT for all samples
    rng = np.random.default_rng(42)
    gt_cache: dict[str, np.ndarray] = {}
    t = time.time()
    for r in rows:
        if not os.path.exists(r["point_cloud"]):
            continue
        gt_cache[r["object_id"]] = load_gt_pts(r["point_cloud"], args.n_eval, rng)
    print(f"GT cached: {len(gt_cache)} samples in {time.time()-t:.1f}s", file=sys.stderr)

    # Per-method aggregation (include teacher too — method may be null but predictions exist)
    methods = [m["name"] for m in cfg["models"]]
    summary = []
    per_sample_rows = []

    for method in methods:
        m_dir = norm_root / method
        if not m_dir.exists():
            print(f"  {method}: no data on disk, skip", file=sys.stderr)
            continue
        sample_dirs = sorted([d for d in m_dir.iterdir() if d.is_dir()])
        cds = []
        f1s = []
        f2s = []
        n_used = 0
        for sd in sample_dirs:
            oid = sd.name
            if oid not in gt_cache:
                continue
            align_f = sd / "alignment_track_a.json"
            eval_f = sd / "eval_pts.npz"
            if not align_f.exists() or not eval_f.exists():
                continue

            pred = load_eval_pts(eval_f)
            align = load_alignment(align_f)
            pred_aligned = apply_alignment(pred, align)
            gt = gt_cache[oid]

            try:
                metrics = compute_geometry_metrics(
                    pred_aligned, gt,
                    f_thresholds=tuple(f_thresholds),
                )
            except Exception as e:
                print(f"  {method}/{oid}: metric fail {e}", file=sys.stderr)
                continue

            cd = float(metrics.chamfer_distance) * 1000  # ×10⁻³ for comparability
            cds.append(cd)
            f1s.append(float(metrics.f_score_001))
            f2s.append(float(metrics.f_score_002))
            n_used += 1
            per_sample_rows.append({
                "model": method, "object_id": oid,
                "chamfer_distance_x1e3": f"{cd:.4f}",
                f"f_score_{f_thresholds[0]}": f"{f1s[-1]:.4f}",
                f"f_score_{f_thresholds[1]}": f"{f2s[-1]:.4f}",
            })

        if n_used == 0:
            print(f"  {method}: 0/0 samples — skip", file=sys.stderr)
            continue

        cd_mean = np.mean(cds)
        cd_median = np.median(cds)
        cd_std = np.std(cds)
        summary.append({
            "model": method, "n": n_used,
            "cd_mean_x1e3": f"{cd_mean:.4f}",
            "cd_median_x1e3": f"{cd_median:.4f}",
            "cd_std_x1e3": f"{cd_std:.4f}",
            f"f{f_thresholds[0]}_mean": f"{np.mean(f1s):.4f}",
            f"f{f_thresholds[1]}_mean": f"{np.mean(f2s):.4f}",
        })
        print(f"  {method}: n={n_used} CD_mean={cd_mean:.4f} CD_median={cd_median:.4f}",
              file=sys.stderr)

    # Write
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"\nSummary written: {args.out_csv}", file=sys.stderr)

    if args.per_sample_csv:
        with open(args.per_sample_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_sample_rows[0].keys()))
            w.writeheader()
            w.writerows(per_sample_rows)
        print(f"Per-sample written: {args.per_sample_csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
