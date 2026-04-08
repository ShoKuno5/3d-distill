#!/usr/bin/env python3
"""Distillation fidelity evaluation: teacher vs student mesh comparison.

Compares student model outputs against a reference model (teacher) using
geometry metrics (CD, F-score, Hausdorff). No ICP alignment — both models
generate from the same input conditioning, so outputs share the same
coordinate frame. Only unit-sphere normalization is applied.

Usage:
    cd /mnt/workspace/kuno/distillation

    # Evaluate all distilled models against teacher
    python distill_methods/scripts/eval_distill.py \
        --config distill_methods/config.yaml \
        --run-dir results/distill_methods/runs/20260406_1131

    # Evaluate specific models
    python distill_methods/scripts/eval_distill.py \
        --config distill_methods/config.yaml \
        --run-dir results/distill_methods/runs/20260406_1131 \
        --models pd_6step cd_4step

    # Override reference model
    python distill_methods/scripts/eval_distill.py \
        --config distill_methods/config.yaml \
        --run-dir results/distill_methods/runs/20260406_1131 \
        --reference flashvdm
"""

import argparse
import csv
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
import trimesh
import yaml

# Add pipeline src to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_DIR / "pipeline"))

from src.geometry.normalize import normalize_to_unit_sphere
from src.geometry.cleaning import clean_mesh
from src.geometry.sampling import sample_surface
from src.evaluation.metrics import (
    compute_geometry_metrics,
    bootstrap_ci,
)

logger = logging.getLogger("eval_distill")


def load_mesh_safe(mesh_path: str) -> trimesh.Trimesh:
    """Load mesh, handle Scene vs Trimesh."""
    mesh = trimesh.load(mesh_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    return mesh


def process_mesh(mesh_path: str, clean_cfg: dict, n_points: int) -> np.ndarray | None:
    """Load, clean, sample, and normalize a mesh. Returns normalized points or None."""
    if not os.path.exists(mesh_path):
        return None

    mesh = load_mesh_safe(mesh_path)
    mesh, report = clean_mesh(
        mesh,
        remove_nan=clean_cfg["remove_nan"],
        remove_inf=clean_cfg["remove_inf"],
        remove_degenerate=clean_cfg["remove_degenerate_faces"],
        remove_unreferenced=clean_cfg["remove_unreferenced_vertices"],
        remove_tiny_components=clean_cfg["remove_tiny_components"],
        tiny_threshold=clean_cfg["tiny_component_threshold"],
    )
    if report.is_empty:
        return None

    pts = sample_surface(mesh, n_points, seed=0)
    pts_norm, _ = normalize_to_unit_sphere(pts)
    return pts_norm


def evaluate_pair(
    object_id: str,
    category: str,
    model_name: str,
    student_mesh_path: str,
    ref_pts: np.ndarray,
    clean_cfg: dict,
    n_points: int,
) -> dict:
    """Evaluate a single (sample, model) pair against reference points."""
    result = {
        "object_id": object_id,
        "category": category,
        "model": model_name,
        "status": "success",
        "error": None,
    }

    try:
        student_pts = process_mesh(student_mesh_path, clean_cfg, n_points)
        if student_pts is None:
            result["status"] = "failed"
            result["error"] = "missing_or_empty"
            return result

        metrics = compute_geometry_metrics(student_pts, ref_pts)
        result.update(asdict(metrics))

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser(description="Distillation fidelity evaluation")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--run-dir", required=True, help="Run directory with predictions/")
    parser.add_argument("--reference", default="teacher_50step",
                        help="Reference model name (default: teacher_50step)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Student models to evaluate (default: all non-reference)")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel workers (default: CPU count)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Load test manifest
    sys.path.insert(0, str(PROJECT_DIR / "pipeline"))
    from src.utils.inference_config import load_and_filter_samples
    cfg_for_samples = dict(cfg)
    cfg_for_samples["output_root"] = args.run_dir
    samples = load_and_filter_samples(
        cfg_for_samples,
        max_samples_override=args.max_samples,
        manifest_key="test_manifest",
    )
    logger.info("Evaluating %d samples", len(samples))

    # Resolve model list
    all_models = cfg["models"]
    ref_cfg = next((m for m in all_models if m["name"] == args.reference), None)
    if ref_cfg is None:
        logger.error("Reference model '%s' not found in config", args.reference)
        sys.exit(1)

    if args.models:
        student_cfgs = [m for m in all_models if m["name"] in args.models]
    else:
        student_cfgs = [m for m in all_models if m["name"] != args.reference]

    student_names = [m["name"] for m in student_cfgs]
    logger.info("Reference: %s", args.reference)
    logger.info("Students: %s", student_names)

    # Paths
    pred_root = os.path.join(args.run_dir, "predictions")
    ref_mesh_fn = ref_cfg["mesh_filename"]
    clean_cfg = cfg["cleaning"]
    n_points = cfg["sampling"]["eval_points"]

    # Prepare reference points per sample
    logger.info("Loading reference meshes (%s)...", args.reference)
    ref_data = {}  # object_id -> normalized points
    ref_failures = []
    for s in samples:
        ref_path = os.path.join(pred_root, args.reference, "default", s.object_id, ref_mesh_fn)
        pts = process_mesh(ref_path, clean_cfg, n_points)
        if pts is not None:
            ref_data[s.object_id] = pts
        else:
            ref_failures.append(s.object_id)
            logger.warning("  Reference missing/empty: %s", s.object_id)

    logger.info("Reference loaded: %d/%d samples", len(ref_data), len(samples))
    if ref_failures:
        logger.warning("  Missing: %s", ref_failures)

    # Build jobs
    jobs = []
    for s in samples:
        if s.object_id not in ref_data:
            continue
        for mcfg in student_cfgs:
            mesh_path = os.path.join(
                pred_root, mcfg["name"], "default", s.object_id, mcfg["mesh_filename"]
            )
            jobs.append((s.object_id, s.category, mcfg["name"], mesh_path))

    n_workers = args.workers or min(os.cpu_count() or 4, len(jobs))
    n_workers = max(1, n_workers)
    logger.info("Running %d evaluations with %d workers", len(jobs), n_workers)

    # Run evaluation
    t0 = time.time()
    results = []

    if n_workers == 1:
        for oid, cat, model, mesh_path in jobs:
            r = evaluate_pair(oid, cat, model, mesh_path, ref_data[oid], clean_cfg, n_points)
            results.append(r)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {}
            for oid, cat, model, mesh_path in jobs:
                f = executor.submit(
                    evaluate_pair, oid, cat, model, mesh_path,
                    ref_data[oid], clean_cfg, n_points,
                )
                futures[f] = (oid, model)

            for i, f in enumerate(as_completed(futures), 1):
                oid, model = futures[f]
                try:
                    results.append(f.result())
                except Exception as e:
                    logger.error("Worker crash %s/%s: %s", oid, model, e)
                    results.append({
                        "object_id": oid, "category": "", "model": model,
                        "status": "failed", "error": str(e),
                    })
                if i % 20 == 0 or i == len(jobs):
                    logger.info("  Progress: %d/%d", i, len(jobs))

    elapsed = time.time() - t0
    logger.info("Evaluation completed in %.1fs", elapsed)

    # Separate successes and failures
    successes = [r for r in results if r["status"] == "success"]
    failures = [r for r in results if r["status"] == "failed"]

    if failures:
        logger.warning("%d failures:", len(failures))
        for f in failures:
            logger.warning("  %s/%s: %s", f["object_id"], f["model"], f["error"])

    # Write per-sample CSV
    metrics_dir = os.path.join(args.run_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    fieldnames = [
        "object_id", "category", "model",
        "chamfer_distance", "chamfer_pred_to_gt", "chamfer_gt_to_pred",
        "hausdorff", "f_score_001", "f_score_002",
    ]

    per_sample_path = os.path.join(metrics_dir, "distill_per_sample.csv")
    with open(per_sample_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(successes)
    logger.info("Per-sample: %s (%d rows)", per_sample_path, len(successes))

    # Write summary
    summary_rows = []
    metric_keys = ["chamfer_distance", "hausdorff", "f_score_001", "f_score_002"]
    boot_cfg = cfg.get("metrics", {}).get("bootstrap_ci", {})

    for model in student_names:
        model_rows = [r for r in successes if r["model"] == model]
        if not model_rows:
            continue

        row = {
            "model": model,
            "reference": args.reference,
            "n_samples": len(model_rows),
        }
        for key in metric_keys:
            vals = np.array([r[key] for r in model_rows if r.get(key) is not None])
            if len(vals) > 0:
                row[f"{key}_mean"] = float(vals.mean())
                row[f"{key}_median"] = float(np.median(vals))
                row[f"{key}_std"] = float(vals.std())
                if boot_cfg.get("enabled", True) and len(vals) >= 3:
                    _, ci_lo, ci_hi = bootstrap_ci(
                        vals,
                        n_bootstrap=boot_cfg.get("n_bootstrap", 1000),
                        confidence=boot_cfg.get("confidence", 0.95),
                    )
                    row[f"{key}_ci95_lo"] = ci_lo
                    row[f"{key}_ci95_hi"] = ci_hi

        n_total = len([s for s in samples if s.object_id in ref_data])
        row["n_failures"] = n_total - len(model_rows)
        summary_rows.append(row)

    if summary_rows:
        summary_path = os.path.join(metrics_dir, "distill_summary.csv")
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()), extrasaction="ignore")
            w.writeheader()
            w.writerows(summary_rows)
        logger.info("Summary: %s", summary_path)

        # Print summary table
        print(f"\n{'='*80}")
        print(f"DISTILLATION FIDELITY — reference: {args.reference}")
        print(f"{'='*80}")
        print(f"{'Model':<16} {'CD (x1e3)':<12} {'HD':<10} {'F@1%':<10} {'F@2%':<10} {'N':>5}")
        print("-" * 65)
        for r in summary_rows:
            cd = r.get("chamfer_distance_mean", float("nan"))
            hd = r.get("hausdorff_mean", float("nan"))
            f1 = r.get("f_score_001_mean", float("nan"))
            f2 = r.get("f_score_002_mean", float("nan"))
            n = r["n_samples"]
            print(f"{r['model']:<16} {cd*1000:<12.4f} {hd:<10.4f} {f1:<10.4f} {f2:<10.4f} {n:>5}")
        print(f"{'='*80}")

    # Write failures
    if failures:
        fail_path = os.path.join(metrics_dir, "distill_failures.csv")
        with open(fail_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["object_id", "category", "model", "error"], extrasaction="ignore")
            w.writeheader()
            w.writerows(failures)


if __name__ == "__main__":
    main()
