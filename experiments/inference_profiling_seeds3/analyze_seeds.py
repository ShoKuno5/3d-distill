#!/usr/bin/env python3
"""Cross-seed aggregation and variance analysis.

Reads per-seed evaluation results and profiling data, produces:
  - per_sample_all_seeds.csv: all seeds x all samples metrics
  - summary_cross_seed.csv: model x track grand mean +/- across-seed std
  - seed_variance_per_object.csv: per-object CD/F-score mean, std, CV
  - seed_variance_per_category.csv: per-category cross-seed stats
  - timing_cross_seed.csv: block timing seed-level variance

Usage:
    python analyze_seeds.py \
        --config config.yaml \
        --eval-dirs eval_seed_0 eval_seed_1 eval_seed_2 \
        --predictions-root predictions/ \
        --seeds 0 1 2 \
        --output-dir aggregated/
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import yaml


def read_csv(path):
    """Read a CSV file and return list of dicts."""
    with open(path) as f:
        return list(csv.DictReader(f))


def compute_stats(values):
    """Compute mean, std, CV for a list of floats."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "cv": 0.0, "n": 0}
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n if n > 1 else 0.0
    std = variance ** 0.5
    cv = std / abs(mean) if abs(mean) > 1e-12 else 0.0
    return {"mean": round(mean, 6), "std": round(std, 6), "cv": round(cv, 4), "n": n}


def write_csv(path, rows, fieldnames=None):
    """Write a list of dicts to CSV."""
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Written: {path} ({len(rows)} rows)")


def collect_per_sample(eval_dirs, seeds):
    """Collect per_sample.csv from each eval dir, tagging with seed."""
    all_rows = []
    for seed, eval_dir in zip(seeds, eval_dirs):
        per_sample_path = os.path.join(eval_dir, "metrics", "per_sample.csv")
        if not os.path.exists(per_sample_path):
            print(f"  WARNING: {per_sample_path} not found, skipping seed {seed}")
            continue
        rows = read_csv(per_sample_path)
        for row in rows:
            row["seed"] = seed
        all_rows.extend(rows)
    return all_rows


def build_summary_cross_seed(all_rows, metric_cols):
    """Compute model x track grand mean +/- across-seed std."""
    # Group by (model, track, metric) -> {seed -> [values]}
    grouped = defaultdict(lambda: defaultdict(list))
    for row in all_rows:
        model = row.get("model", "")
        track = row.get("track", "")
        seed = row["seed"]
        for col in metric_cols:
            if col in row and row[col]:
                try:
                    val = float(row[col])
                    grouped[(model, track, col)][seed].append(val)
                except ValueError:
                    pass

    summary_rows = []
    for (model, track, metric), seed_vals in sorted(grouped.items()):
        # Compute per-seed means
        seed_means = []
        for seed, vals in sorted(seed_vals.items()):
            if vals:
                seed_means.append(sum(vals) / len(vals))

        stats = compute_stats(seed_means)
        summary_rows.append({
            "model": model,
            "track": track,
            "metric": metric,
            "grand_mean": stats["mean"],
            "across_seed_std": stats["std"],
            "cv": stats["cv"],
            "n_seeds": stats["n"],
        })

    return summary_rows


def build_variance_per_object(all_rows, metric_cols):
    """Per-object seed variance for key metrics."""
    # Group by (model, track, object_id, metric) -> [values across seeds]
    grouped = defaultdict(list)
    categories = {}
    for row in all_rows:
        model = row.get("model", "")
        track = row.get("track", "")
        oid = row.get("object_id", "")
        cat = row.get("category", "")
        if oid and cat:
            categories[oid] = cat
        for col in metric_cols:
            if col in row and row[col]:
                try:
                    grouped[(model, track, oid, col)].append(float(row[col]))
                except ValueError:
                    pass

    result_rows = []
    for (model, track, oid, metric), vals in sorted(grouped.items()):
        stats = compute_stats(vals)
        result_rows.append({
            "model": model,
            "track": track,
            "object_id": oid,
            "category": categories.get(oid, ""),
            "metric": metric,
            "mean": stats["mean"],
            "std": stats["std"],
            "cv": stats["cv"],
            "n_seeds": stats["n"],
        })

    return result_rows


def build_variance_per_category(object_rows):
    """Aggregate object-level variance by category."""
    # Group by (model, track, category, metric) -> list of object CVs
    grouped = defaultdict(list)
    for row in object_rows:
        key = (row["model"], row["track"], row["category"], row["metric"])
        grouped[key].append({
            "mean": row["mean"],
            "std": row["std"],
            "cv": row["cv"],
        })

    result_rows = []
    for (model, track, cat, metric), obj_stats in sorted(grouped.items()):
        means = [o["mean"] for o in obj_stats]
        stds = [o["std"] for o in obj_stats]
        cvs = [o["cv"] for o in obj_stats]
        result_rows.append({
            "model": model,
            "track": track,
            "category": cat,
            "metric": metric,
            "n_objects": len(obj_stats),
            "mean_of_means": round(sum(means) / len(means), 6) if means else 0,
            "mean_std": round(sum(stds) / len(stds), 6) if stds else 0,
            "mean_cv": round(sum(cvs) / len(cvs), 4) if cvs else 0,
            "max_cv": round(max(cvs), 4) if cvs else 0,
        })

    return result_rows


def build_timing_cross_seed(predictions_root, model_names, seeds):
    """Analyze timing variance across seeds from profile.json files."""
    # Group by (model, object_id, block) -> [wall_sec across seeds]
    block_values = defaultdict(list)
    total_values = defaultdict(list)

    for model in model_names:
        for seed in seeds:
            seed_dir = os.path.join(predictions_root, model, f"seed_{seed}")
            if not os.path.isdir(seed_dir):
                print(f"  WARNING: {seed_dir} not found")
                continue
            for oid in sorted(os.listdir(seed_dir)):
                profile_path = os.path.join(seed_dir, oid, "profile.json")
                if not os.path.exists(profile_path):
                    continue
                with open(profile_path) as f:
                    profile = json.load(f)
                if profile.get("status") == "failed":
                    continue
                for block_name, block_data in profile.get("blocks", {}).items():
                    wall_sec = block_data.get("wall_sec", 0.0)
                    block_values[(model, oid, block_name)].append(wall_sec)
                total_values[(model, oid)].append(profile.get("total_sec", 0.0))

    # Summarize per (model, block) across objects and seeds
    block_summary = defaultdict(lambda: {"means": [], "stds": [], "cvs": []})
    for (model, oid, block), vals in block_values.items():
        stats = compute_stats(vals)
        block_summary[(model, block)]["means"].append(stats["mean"])
        block_summary[(model, block)]["stds"].append(stats["std"])
        block_summary[(model, block)]["cvs"].append(stats["cv"])

    # Also total
    total_summary = defaultdict(lambda: {"means": [], "stds": [], "cvs": []})
    for (model, oid), vals in total_values.items():
        stats = compute_stats(vals)
        total_summary[model]["means"].append(stats["mean"])
        total_summary[model]["stds"].append(stats["std"])
        total_summary[model]["cvs"].append(stats["cv"])

    result_rows = []
    for (model, block), agg in sorted(block_summary.items()):
        n = len(agg["means"])
        result_rows.append({
            "model": model,
            "block": block,
            "n_objects": n,
            "mean_time": round(sum(agg["means"]) / n, 4) if n else 0,
            "mean_cross_seed_std": round(sum(agg["stds"]) / n, 4) if n else 0,
            "mean_cross_seed_cv": round(sum(agg["cvs"]) / n, 4) if n else 0,
            "max_cross_seed_cv": round(max(agg["cvs"]), 4) if agg["cvs"] else 0,
        })

    for model, agg in sorted(total_summary.items()):
        n = len(agg["means"])
        result_rows.append({
            "model": model,
            "block": "TOTAL",
            "n_objects": n,
            "mean_time": round(sum(agg["means"]) / n, 4) if n else 0,
            "mean_cross_seed_std": round(sum(agg["stds"]) / n, 4) if n else 0,
            "mean_cross_seed_cv": round(sum(agg["cvs"]) / n, 4) if n else 0,
            "max_cross_seed_cv": round(max(agg["cvs"]), 4) if agg["cvs"] else 0,
        })

    return result_rows


def main():
    parser = argparse.ArgumentParser(description="Cross-seed aggregation and variance analysis")
    parser.add_argument("--config", required=True, help="Base config YAML (for model names)")
    parser.add_argument("--eval-dirs", nargs="+", required=True,
                        help="Eval output directories (one per seed)")
    parser.add_argument("--predictions-root", required=True,
                        help="Root predictions directory containing model/seed_N/ subdirs")
    parser.add_argument("--seeds", nargs="+", type=int, required=True,
                        help="Seed numbers corresponding to eval-dirs")
    parser.add_argument("--output-dir", required=True, help="Output directory for aggregated CSVs")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_names = [m["name"] for m in cfg["models"]]
    os.makedirs(args.output_dir, exist_ok=True)

    # Key metrics to track
    metric_cols = [
        "chamfer_distance", "f_score_0.01", "f_score_0.02",
        "hausdorff_distance", "alignment_scale",
    ]

    print("=== Cross-seed aggregation ===")

    # 1. Collect all per-sample data
    print("\n[1/5] Collecting per-sample metrics across seeds...")
    all_rows = collect_per_sample(args.eval_dirs, args.seeds)
    write_csv(
        os.path.join(args.output_dir, "per_sample_all_seeds.csv"),
        all_rows,
    )

    # 2. Summary cross-seed
    print("\n[2/5] Computing cross-seed summary (model x track)...")
    summary_rows = build_summary_cross_seed(all_rows, metric_cols)
    write_csv(
        os.path.join(args.output_dir, "summary_cross_seed.csv"),
        summary_rows,
    )

    # 3. Per-object variance
    print("\n[3/5] Computing per-object seed variance...")
    object_rows = build_variance_per_object(all_rows, metric_cols)
    write_csv(
        os.path.join(args.output_dir, "seed_variance_per_object.csv"),
        object_rows,
    )

    # 4. Per-category variance
    print("\n[4/5] Computing per-category seed variance...")
    category_rows = build_variance_per_category(object_rows)
    write_csv(
        os.path.join(args.output_dir, "seed_variance_per_category.csv"),
        category_rows,
    )

    # 5. Timing cross-seed
    print("\n[5/5] Analyzing timing variance across seeds...")
    timing_rows = build_timing_cross_seed(args.predictions_root, model_names, args.seeds)
    write_csv(
        os.path.join(args.output_dir, "timing_cross_seed.csv"),
        timing_rows,
    )

    print("\n=== Aggregation complete ===")
    print(f"Results in: {args.output_dir}")


if __name__ == "__main__":
    main()
