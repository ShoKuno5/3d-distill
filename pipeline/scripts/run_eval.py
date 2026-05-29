#!/usr/bin/env python3
"""Aligned geometry evaluation pipeline for Toys4k.

Runs Track A (similarity ICP) and Track B (scale only) evaluation
for all models on the specified sample subset.

Supports parallel evaluation via --workers N (default: number of CPUs).

Usage:
    python pipeline/scripts/run_eval.py \
        --config experiments/toys4k_baseline/config.yaml

    # Override subset size:
    python pipeline/scripts/run_eval.py \
        --config experiments/toys4k_baseline/config.yaml \
        --max-samples 5

    # Parallel with 8 workers:
    python pipeline/scripts/run_eval.py \
        --config experiments/toys4k_baseline/config.yaml \
        --workers 8
"""

import argparse
import csv
import json
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

# Add src to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.data.toys4k import (
    load_manifest,
    filter_samples,
    load_gt_pointcloud,
    save_sample_ids,
)
from src.geometry.normalize import normalize_to_unit_sphere, NormTransform
from src.geometry.cleaning import clean_mesh
from src.geometry.sampling import sample_surface
from src.geometry.alignment import (
    align_similarity_icp,
    align_scale_only,
    apply_alignment,
)
from src.evaluation.metrics import (
    compute_geometry_metrics,
    compute_mesh_quality,
    bootstrap_ci,
    GeometryMetrics,
    MeshQualityStats,
)


def setup_logging(log_dir: str, level: str = "INFO") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("eval")
    logger.setLevel(getattr(logging, level))
    # File handler
    fh = logging.FileHandler(os.path.join(log_dir, "eval.log"), mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, level))
    ch.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(ch)
    return logger


def load_config(config_path: str) -> dict:
    from src.utils.inference_config import resolve_model_paths
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return resolve_model_paths(cfg)


def load_mesh_safe(mesh_path: str) -> trimesh.Trimesh:
    """Load mesh, handle Scene vs Trimesh."""
    mesh = trimesh.load(mesh_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    return mesh


def save_pointcloud(pts: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, points=pts)


def save_transform(result, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "transform_4x4": result.transform_4x4.tolist(),
        "rotation": result.rotation.tolist(),
        "translation": result.translation.tolist(),
        "scale": float(result.scale),
        "chamfer_cost": float(result.chamfer_cost),
        "initial_rotation_idx": int(result.initial_rotation_idx),
        "n_icp_iterations": int(result.n_icp_iterations),
        "method": result.method,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def evaluate_one(
    sid: str,
    cat: str,
    model_name: str,
    mesh_path: str,
    gt_pts_norm: np.ndarray,
    output_root: str,
    clean_cfg: dict,
    align_cfg: dict,
    n_align_pts: int,
    n_eval_pts: int,
    skip_track_b: bool,
) -> dict:
    """Evaluate a single (sample, model) pair. Returns a dict with results.

    This function is designed to be called from a process pool.
    """
    result = {
        "object_id": sid,
        "category": cat,
        "model": model_name,
        "per_sample_rows": [],
        "mesh_quality_row": None,
        "timing_row": None,
        "failure": None,
    }

    if not os.path.exists(mesh_path):
        result["failure"] = {
            "object_id": sid, "category": cat, "model": model_name,
            "error": "prediction_not_found",
        }
        return result

    t_total_start = time.time()

    try:
        # --- Load and clean mesh ---
        t_load_start = time.time()
        raw_mesh = load_mesh_safe(mesh_path)
        t_load = time.time() - t_load_start

        # Save raw prediction reference (symlink to avoid duplication)
        raw_pred_dir = os.path.join(output_root, "raw_predictions", model_name, sid)
        os.makedirs(raw_pred_dir, exist_ok=True)
        raw_ref_path = os.path.join(raw_pred_dir, "mesh_raw.obj")
        if not os.path.exists(raw_ref_path):
            try:
                os.symlink(os.path.abspath(mesh_path), raw_ref_path)
            except FileExistsError:
                pass  # race condition with parallel workers

        # Clean mesh
        t_clean_start = time.time()
        cleaned_mesh, clean_report = clean_mesh(
            raw_mesh,
            remove_nan=clean_cfg["remove_nan"],
            remove_inf=clean_cfg["remove_inf"],
            remove_degenerate=clean_cfg["remove_degenerate_faces"],
            remove_unreferenced=clean_cfg["remove_unreferenced_vertices"],
            remove_tiny_components=clean_cfg["remove_tiny_components"],
            tiny_threshold=clean_cfg["tiny_component_threshold"],
        )
        t_clean = time.time() - t_clean_start

        if clean_report.is_empty:
            result["failure"] = {
                "object_id": sid, "category": cat, "model": model_name,
                "error": "empty_after_cleaning",
            }
            return result

        # Save cleaned mesh
        norm_pred_dir = os.path.join(output_root, "normalized_predictions", model_name, sid)
        os.makedirs(norm_pred_dir, exist_ok=True)
        cleaned_mesh.export(os.path.join(norm_pred_dir, "mesh_cleaned.obj"))

        # --- Mesh quality stats ---
        mq = compute_mesh_quality(cleaned_mesh)
        result["mesh_quality_row"] = {
            "object_id": sid, "category": cat, "model": model_name,
            **asdict(mq),
        }

        # --- Sample point clouds ---
        t_sample_start = time.time()
        align_pts = sample_surface(cleaned_mesh, n_align_pts, seed=0)
        eval_pts = sample_surface(cleaned_mesh, n_eval_pts, seed=0)
        t_sample = time.time() - t_sample_start

        # --- Normalize prediction ---
        align_pts_norm, pred_transform = normalize_to_unit_sphere(align_pts)
        eval_pts_norm = (eval_pts - pred_transform.center) / pred_transform.scale

        # Save sampled point cloud
        save_pointcloud(eval_pts_norm, os.path.join(norm_pred_dir, "eval_pts.npz"))

        # === Track A: Similarity ICP ===
        t_align_start = time.time()
        scale_clamp = align_cfg["track_a"].get("scale_clamp")
        icp_kwargs = dict(
            max_iterations=align_cfg["track_a"]["icp_max_iterations"],
            n_initial_rotations=align_cfg["track_a"]["initial_rotations"],
        )
        if scale_clamp is not None:
            icp_kwargs["scale_clamp"] = tuple(scale_clamp)
        track_a_result = align_similarity_icp(
            align_pts_norm,
            gt_pts_norm,
            **icp_kwargs,
        )
        t_align = time.time() - t_align_start

        # Apply alignment to eval points
        eval_pts_aligned = apply_alignment(eval_pts_norm, track_a_result)

        # Save alignment transform
        save_transform(
            track_a_result,
            os.path.join(norm_pred_dir, "alignment_track_a.json"),
        )

        # Compute Track A metrics
        t_metrics_start = time.time()
        track_a_metrics = compute_geometry_metrics(eval_pts_aligned, gt_pts_norm)
        t_metrics = time.time() - t_metrics_start

        t_total = time.time() - t_total_start

        row_a = {
            "object_id": sid,
            "category": cat,
            "model": model_name,
            "track": "A",
            "chamfer_distance": track_a_metrics.chamfer_distance,
            "chamfer_pred_to_gt": track_a_metrics.chamfer_pred_to_gt,
            "chamfer_gt_to_pred": track_a_metrics.chamfer_gt_to_pred,
            "hausdorff": track_a_metrics.hausdorff,
            "f_score_001": track_a_metrics.f_score_001,
            "f_score_002": track_a_metrics.f_score_002,
            "alignment_cost": track_a_result.chamfer_cost,
            "alignment_scale": track_a_result.scale,
            "alignment_rot_idx": track_a_result.initial_rotation_idx,
        }
        result["per_sample_rows"].append(row_a)

        # Log line (formatted for main process to print)
        result["log_a"] = (
            f"    Track A: CD={track_a_metrics.chamfer_distance:.6f} "
            f"F@1%={track_a_metrics.f_score_001:.4f} "
            f"F@2%={track_a_metrics.f_score_002:.4f} "
            f"HD={track_a_metrics.hausdorff:.4f} "
            f"align_cost={track_a_result.chamfer_cost:.6f} "
            f"({t_align:.1f}s align, {t_metrics:.1f}s metrics)"
        )

        # === Track B: Scale only ===
        if not skip_track_b:
            track_b_result = align_scale_only(eval_pts_norm, gt_pts_norm)
            track_b_metrics = compute_geometry_metrics(eval_pts_norm, gt_pts_norm)

            save_transform(
                track_b_result,
                os.path.join(norm_pred_dir, "alignment_track_b.json"),
            )

            row_b = {
                "object_id": sid,
                "category": cat,
                "model": model_name,
                "track": "B",
                "chamfer_distance": track_b_metrics.chamfer_distance,
                "chamfer_pred_to_gt": track_b_metrics.chamfer_pred_to_gt,
                "chamfer_gt_to_pred": track_b_metrics.chamfer_gt_to_pred,
                "hausdorff": track_b_metrics.hausdorff,
                "f_score_001": track_b_metrics.f_score_001,
                "f_score_002": track_b_metrics.f_score_002,
                "alignment_cost": track_b_result.chamfer_cost,
                "alignment_scale": track_b_result.scale,
                "alignment_rot_idx": -1,
            }
            result["per_sample_rows"].append(row_b)

            result["log_b"] = (
                f"    Track B: CD={track_b_metrics.chamfer_distance:.6f} "
                f"F@1%={track_b_metrics.f_score_001:.4f} "
                f"F@2%={track_b_metrics.f_score_002:.4f}"
            )

        # --- Timing ---
        result["timing_row"] = {
            "object_id": sid,
            "category": cat,
            "model": model_name,
            "load_sec": round(t_load, 3),
            "clean_sec": round(t_clean, 3),
            "sample_sec": round(t_sample, 3),
            "align_sec": round(t_align, 3),
            "metrics_sec": round(t_metrics, 3),
            "total_sec": round(t_total, 3),
        }

    except Exception as e:
        import traceback
        result["failure"] = {
            "object_id": sid, "category": cat, "model": model_name,
            "error": str(e),
        }
        result["traceback"] = traceback.format_exc()

    return result


def main():
    parser = argparse.ArgumentParser(description="Aligned geometry evaluation")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Override max_samples from config")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Override model list from config")
    parser.add_argument("--skip-track-b", action="store_true",
                        help="Skip Track B evaluation")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: CPU count)")
    parser.add_argument("--scale-clamp-min", type=float, default=None,
                        help="Override Track A scale clamp lower bound (default from config)")
    parser.add_argument("--scale-clamp-max", type=float, default=None,
                        help="Override Track A scale clamp upper bound (default from config)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_root = cfg["output_root"]

    logger = setup_logging(
        os.path.join(output_root, "logs"),
        cfg.get("execution", {}).get("log_level", "INFO"),
    )
    logger.info(f"Config loaded from {args.config}")

    # --- Load samples ---
    max_samples = args.max_samples or cfg["dataset"].get("max_samples")
    manifest_path = cfg["dataset"].get("test_manifest") or cfg["dataset"]["manifest"]
    samples = load_manifest(manifest_path)
    samples = filter_samples(
        samples,
        max_samples=max_samples,
        sample_ids_file=cfg["dataset"].get("sample_ids_file"),
        category_filter=cfg["dataset"].get("category_filter"),
    )
    logger.info(f"Evaluating {len(samples)} samples")

    # Save sample IDs
    save_sample_ids(samples, os.path.join(output_root, "cache", "sample_ids.txt"))

    # --- Model configs ---
    model_cfgs = cfg["models"]
    if args.models:
        model_cfgs = [m for m in model_cfgs if m["name"] in args.models]
    model_names = [m["name"] for m in model_cfgs]

    # --- Sampling / cleaning config ---
    n_align_pts = cfg["sampling"]["alignment_points"]
    n_eval_pts = cfg["sampling"]["eval_points"]
    clean_cfg = cfg["cleaning"]
    align_cfg = cfg["alignment"]
    continue_on_failure = cfg.get("execution", {}).get("continue_on_failure", True)

    # CLI override: scale_clamp for Track A (fairness sensitivity runs)
    if args.scale_clamp_min is not None or args.scale_clamp_max is not None:
        current = align_cfg["track_a"].get("scale_clamp", [0.5, 2.0])
        s_min = args.scale_clamp_min if args.scale_clamp_min is not None else current[0]
        s_max = args.scale_clamp_max if args.scale_clamp_max is not None else current[1]
        align_cfg["track_a"]["scale_clamp"] = [s_min, s_max]
        logger.info(f"Track A scale_clamp overridden to [{s_min}, {s_max}]")

    # --- Determine worker count ---
    n_workers = args.workers
    if n_workers is None:
        n_workers = min(os.cpu_count() or 4, len(samples) * len(model_cfgs))
    n_workers = max(1, n_workers)

    # --- Results storage ---
    per_sample_rows = []
    mesh_quality_rows = []
    timing_rows = []
    failure_rows = []

    metrics_dir = os.path.join(output_root, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    # --- Prepare GT point clouds (sequential, fast) ---
    gt_data = {}  # sid -> gt_pts_norm
    for sample in samples:
        sid = sample.object_id
        cat = sample.category
        t_gt_start = time.time()
        try:
            gt_pts_raw, gt_normals = load_gt_pointcloud(sample.point_cloud)
        except Exception as e:
            logger.error(f"  Cannot load GT for {sid}: {e}")
            failure_rows.append({"object_id": sid, "category": cat, "model": "GT", "error": str(e)})
            continue
        gt_pts_norm, gt_transform = normalize_to_unit_sphere(gt_pts_raw)
        t_gt = time.time() - t_gt_start

        # Cache GT canonical
        gt_cache_path = os.path.join(output_root, "cache", "gt_canonical", f"{sid}.npz")
        save_pointcloud(gt_pts_norm, gt_cache_path)

        gt_data[sid] = gt_pts_norm
        logger.info(f"GT {sid}: {gt_pts_norm.shape[0]} pts ({t_gt:.2f}s)")

    # --- Build job list ---
    jobs = []
    for sample in samples:
        sid = sample.object_id
        if sid not in gt_data:
            continue
        cat = sample.category
        for mcfg in model_cfgs:
            mesh_path = os.path.join(mcfg["predictions_root"], sid, mcfg["mesh_filename"])
            jobs.append((sid, cat, mcfg["name"], mesh_path))

    n_jobs = len(jobs)
    logger.info(f"Running {n_jobs} evaluations with {n_workers} workers")

    # --- Parallel evaluation ---
    t_eval_start = time.time()

    if n_workers == 1:
        # Sequential fallback
        results = []
        for sid, cat, model_name, mesh_path in jobs:
            r = evaluate_one(
                sid, cat, model_name, mesh_path,
                gt_data[sid], output_root, clean_cfg, align_cfg,
                n_align_pts, n_eval_pts, args.skip_track_b,
            )
            results.append(r)
    else:
        futures = {}
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for sid, cat, model_name, mesh_path in jobs:
                future = executor.submit(
                    evaluate_one,
                    sid, cat, model_name, mesh_path,
                    gt_data[sid], output_root, clean_cfg, align_cfg,
                    n_align_pts, n_eval_pts, args.skip_track_b,
                )
                futures[future] = (sid, model_name)

            results = []
            for i, future in enumerate(as_completed(futures), 1):
                sid, model_name = futures[future]
                try:
                    r = future.result()
                    results.append(r)
                except Exception as e:
                    logger.error(f"  Worker crash for {sid}/{model_name}: {e}")
                    results.append({
                        "object_id": sid, "model": model_name,
                        "per_sample_rows": [], "mesh_quality_row": None,
                        "timing_row": None,
                        "failure": {"object_id": sid, "category": "", "model": model_name, "error": str(e)},
                    })

                if i % 10 == 0 or i == n_jobs:
                    logger.info(f"  Progress: {i}/{n_jobs} jobs done")

    t_eval_total = time.time() - t_eval_start

    # --- Collect results ---
    for r in results:
        sid = r["object_id"]
        model_name = r["model"]

        if r.get("failure"):
            failure_rows.append(r["failure"])
            if "traceback" in r:
                logger.debug(f"  {sid}/{model_name} traceback:\n{r['traceback']}")
            logger.warning(f"  {sid}/{model_name}: {r['failure']['error']}")
            if not continue_on_failure and r["failure"]["error"] != "prediction_not_found":
                raise RuntimeError(f"Evaluation failed for {sid}/{model_name}")
            continue

        per_sample_rows.extend(r["per_sample_rows"])
        if r["mesh_quality_row"]:
            mesh_quality_rows.append(r["mesh_quality_row"])
        if r["timing_row"]:
            timing_rows.append(r["timing_row"])

        # Print logs
        if "log_a" in r:
            logger.info(f"  {sid}/{model_name}:")
            logger.info(r["log_a"])
        if "log_b" in r:
            logger.info(r["log_b"])

    logger.info(f"Evaluation completed in {t_eval_total:.1f}s ({n_workers} workers)")

    # --- Write CSV outputs ---
    logger.info("Writing results...")

    # Per-sample metrics
    per_sample_path = os.path.join(metrics_dir, "per_sample.csv")
    _write_csv(per_sample_path, per_sample_rows, [
        "object_id", "category", "model", "track",
        "chamfer_distance", "chamfer_pred_to_gt", "chamfer_gt_to_pred",
        "hausdorff", "f_score_001", "f_score_002",
        "alignment_cost", "alignment_scale", "alignment_rot_idx",
    ])
    logger.info(f"  Per-sample metrics: {per_sample_path} ({len(per_sample_rows)} rows)")

    # Mesh quality
    mq_path = os.path.join(metrics_dir, "mesh_quality.csv")
    if mesh_quality_rows:
        _write_csv(mq_path, mesh_quality_rows, list(mesh_quality_rows[0].keys()))
        logger.info(f"  Mesh quality: {mq_path}")

    # Timing
    timing_path = os.path.join(metrics_dir, "timing.csv")
    if timing_rows:
        _write_csv(timing_path, timing_rows, list(timing_rows[0].keys()))
        logger.info(f"  Timing: {timing_path}")

    # Failures (always write to clear stale entries from previous runs)
    failure_path = os.path.join(metrics_dir, "failures.csv")
    _write_csv(failure_path, failure_rows, ["object_id", "category", "model", "error"])
    if failure_rows:
        logger.info(f"  Failures: {failure_path} ({len(failure_rows)} entries)")
    else:
        logger.info(f"  No failures (cleared {failure_path})")

    # --- Summary ---
    _write_summary(per_sample_rows, model_names, metrics_dir, cfg, logger)

    # --- Multiview rendering (for FD and CLIP-I) ---
    fd_cfg_check = cfg.get("metrics", {}).get("frechet_distance", {})
    clip_i_cfg_check = cfg.get("metrics", {}).get("clip_image", {})
    needs_renders = fd_cfg_check.get("enabled", False) or clip_i_cfg_check.get("enabled", False)
    if needs_renders:
        logger.info("Rendering multiview images (GT + per-model)...")
        try:
            from src.evaluation.multiview_renderer import render_multiview, render_all_meshes
            renders_dir = os.path.join(output_root, "multiview_renders")
            gt_dir = os.path.join(renders_dir, "gt")

            mv_cfg = fd_cfg_check.get("multiview", {})
            azim_cfg = mv_cfg.get("azimuths", [0, 90, 180, 270])
            elev_cfg = mv_cfg.get("elevation", 30)
            res_cfg = mv_cfg.get("resolution", 512)

            # GT renders (one-time, cached). Only FD consumes these — CLIP-I
            # compares the input image to per-model renders, not the GT mesh —
            # so skip GT rendering entirely when FD is disabled.
            if fd_cfg_check.get("enabled", False):
                n_gt_rendered = 0
                for s in samples:
                    sid_dir = os.path.join(gt_dir, s.object_id)
                    if all(
                        os.path.exists(os.path.join(sid_dir, f"view_{int(az)}.png"))
                        for az in azim_cfg
                    ):
                        continue
                    try:
                        render_multiview(s.mesh_obj, sid_dir,
                                         resolution=res_cfg, azimuths=azim_cfg, elevation=elev_cfg)
                        n_gt_rendered += 1
                    except Exception as e:
                        logger.error(f"  GT render failed for {s.object_id}: {e}")
                logger.info(f"  GT renders: {n_gt_rendered} new (cached others)")

            # Per-model renders (needed by both FD and CLIP-I)
            object_ids = [s.object_id for s in samples]
            for mcfg in model_cfgs:
                model_name = mcfg["name"]
                pred_render_dir = os.path.join(renders_dir, model_name)
                n_pred = render_all_meshes(
                    mcfg["predictions_root"],
                    pred_render_dir,
                    mcfg.get("mesh_filename", "mesh_raw.obj"),
                    object_ids,
                    resolution=res_cfg,
                    azimuths=azim_cfg,
                    elevation=elev_cfg,
                )
                logger.info(f"  {model_name}: {n_pred}/{len(object_ids)} rendered")
        except Exception as e:
            logger.error(f"Render orchestration failed: {e}", exc_info=True)

    # --- Frechet Distance (if configured) ---
    fd_cfg = cfg.get("metrics", {}).get("frechet_distance", {})
    if fd_cfg.get("enabled", False):
        logger.info("Computing Frechet Distance metrics...")
        try:
            from src.evaluation.frechet_distance import compute_fd_for_model
            renders_dir = os.path.join(output_root, "multiview_renders")
            gt_dir = os.path.join(renders_dir, "gt")
            fd_models = fd_cfg.get("models", ["inception_v3"])
            fd_rows = []

            for mcfg in model_cfgs:
                model_name = mcfg["name"]
                pred_dir = os.path.join(renders_dir, model_name)
                if not os.path.isdir(pred_dir):
                    logger.warning(f"  No renders for {model_name}, skipping FD")
                    continue
                for feat_model in fd_models:
                    try:
                        fd = compute_fd_for_model(pred_dir, gt_dir, model_name=feat_model)
                        logger.info(f"  {model_name} FD_{feat_model}: {fd:.4f}")
                        fd_rows.append({
                            "model": model_name,
                            "feature_extractor": feat_model,
                            "frechet_distance": fd,
                        })
                    except Exception as e:
                        logger.error(f"  FD failed for {model_name}/{feat_model}: {e}")

            if fd_rows:
                fd_path = os.path.join(metrics_dir, "frechet_distance.csv")
                _write_csv(fd_path, fd_rows,
                           ["model", "feature_extractor", "frechet_distance"])
                logger.info(f"  FD results: {fd_path}")
        except ImportError as e:
            logger.warning(f"  FD computation skipped (missing dependency): {e}")


    # --- CLIP-I (if configured) ---
    clip_i_cfg = cfg.get("metrics", {}).get("clip_image", {})
    if clip_i_cfg.get("enabled", False):
        logger.info("Computing CLIP-I metrics...")
        try:
            from src.evaluation.clip_image_metric import compute_clip_image_for_run
            renders_root = os.path.join(output_root, "multiview_renders")
            azim_cfg = fd_cfg_check.get("multiview", {}).get("azimuths", [0, 90, 180, 270])
            compute_clip_image_for_run(
                cfg=cfg, samples=samples,
                renders_root=renders_root, output_root=output_root,
                model_cfgs=model_cfgs,
                view_names=[f"view_{int(az)}.png" for az in azim_cfg],
            )
        except Exception as e:
            logger.warning(f"  CLIP-I skipped: {e}", exc_info=True)

    # --- Cross-modal (ULIP-I, Uni3D-I) ---
    cm_cfg = cfg.get("metrics", {}).get("cross_modal", {})
    if cm_cfg.get("enabled", False):
        logger.info("Computing ULIP-I and Uni3D-I metrics...")
        try:
            from src.evaluation.cross_modal_metric import compute_cross_modal_for_run
            compute_cross_modal_for_run(
                cfg=cfg, samples=samples, output_root=output_root,
                model_cfgs=model_cfgs,
            )
        except Exception as e:
            logger.warning(f"  Cross-modal skipped: {e}", exc_info=True)

    logger.info("Evaluation complete.")


def _write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(
    per_sample_rows: list[dict],
    model_names: list[str],
    metrics_dir: str,
    cfg: dict,
    logger: logging.Logger,
) -> None:
    """Compute and write summary statistics."""
    metric_keys = [
        "chamfer_distance", "hausdorff", "f_score_001", "f_score_002",
    ]
    summary_rows = []

    for track in ["A", "B"]:
        track_rows = [r for r in per_sample_rows if r.get("track") == track]
        if not track_rows:
            continue

        for model in model_names:
            model_rows = [r for r in track_rows if r["model"] == model]
            if not model_rows:
                continue

            row = {"model": model, "track": track, "n_samples": len(model_rows)}

            for key in metric_keys:
                vals = np.array([r[key] for r in model_rows if r.get(key) is not None])
                if len(vals) > 0:
                    row[f"{key}_mean"] = float(vals.mean())
                    row[f"{key}_median"] = float(np.median(vals))
                    row[f"{key}_std"] = float(vals.std())

                    # Bootstrap CI
                    boot_cfg = cfg.get("metrics", {}).get("bootstrap_ci", {})
                    if boot_cfg.get("enabled", True) and len(vals) >= 3:
                        _, ci_lo, ci_hi = bootstrap_ci(
                            vals,
                            n_bootstrap=boot_cfg.get("n_bootstrap", 1000),
                            confidence=boot_cfg.get("confidence", 0.95),
                        )
                        row[f"{key}_ci95_lo"] = ci_lo
                        row[f"{key}_ci95_hi"] = ci_hi

            # Failure stats
            n_total = len(set(r["object_id"] for r in track_rows))
            n_success = len(model_rows)
            row["n_failures"] = n_total - n_success
            row["failure_rate"] = (n_total - n_success) / n_total if n_total > 0 else 0

            summary_rows.append(row)

    if summary_rows:
        summary_path = os.path.join(metrics_dir, "summary.csv")
        _write_csv(summary_path, summary_rows, list(summary_rows[0].keys()))
        logger.info(f"  Summary: {summary_path}")

        # Print summary table
        print(f"\n{'='*80}")
        print("SUMMARY — Track A (Similarity ICP Aligned)")
        print(f"{'='*80}")
        print(f"{'Model':<12} {'CD (x1e3)':<12} {'HD':<10} {'F@1%':<10} {'F@2%':<10} {'N':>5}")
        print("-" * 60)
        for r in summary_rows:
            if r["track"] != "A":
                continue
            cd = r.get("chamfer_distance_mean", float("nan"))
            hd = r.get("hausdorff_mean", float("nan"))
            f1 = r.get("f_score_001_mean", float("nan"))
            f2 = r.get("f_score_002_mean", float("nan"))
            n = r["n_samples"]
            print(f"{r['model']:<12} {cd*1000:<12.3f} {hd:<10.4f} {f1:<10.4f} {f2:<10.4f} {n:>5}")

        track_b = [r for r in summary_rows if r["track"] == "B"]
        if track_b:
            print(f"\n{'='*80}")
            print("SUMMARY — Track B (Scale Only, No Rotation)")
            print(f"{'='*80}")
            print(f"{'Model':<12} {'CD (x1e3)':<12} {'HD':<10} {'F@1%':<10} {'F@2%':<10} {'N':>5}")
            print("-" * 60)
            for r in track_b:
                cd = r.get("chamfer_distance_mean", float("nan"))
                hd = r.get("hausdorff_mean", float("nan"))
                f1 = r.get("f_score_001_mean", float("nan"))
                f2 = r.get("f_score_002_mean", float("nan"))
                n = r["n_samples"]
                print(f"{r['model']:<12} {cd*1000:<12.3f} {hd:<10.4f} {f1:<10.4f} {f2:<10.4f} {n:>5}")


if __name__ == "__main__":
    main()
