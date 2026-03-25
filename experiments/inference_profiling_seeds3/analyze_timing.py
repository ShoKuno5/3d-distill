#!/usr/bin/env python3
"""Aggregate block-level profiling results from profile.json files.

Reads all profile.json from the predictions directory, computes per-block
statistics (mean, std, median, min, max), and outputs summary tables.

Usage:
    python experiments/inference_profiling/analyze_timing.py \
        --config experiments/inference_profiling/config.yaml
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import yaml


def load_profiles(pred_root):
    """Load all profile.json files from prediction directories."""
    profiles = []
    if not os.path.isdir(pred_root):
        return profiles
    for obj_id in sorted(os.listdir(pred_root)):
        profile_path = os.path.join(pred_root, obj_id, "profile.json")
        if os.path.exists(profile_path):
            with open(profile_path) as f:
                profiles.append(json.load(f))
    return profiles


def compute_stats(values):
    """Compute summary statistics for a list of values."""
    if not values:
        return {}
    n = len(values)
    sorted_v = sorted(values)
    mean = sum(values) / n
    median = sorted_v[n // 2] if n % 2 else (sorted_v[n // 2 - 1] + sorted_v[n // 2]) / 2
    variance = sum((x - mean) ** 2 for x in values) / n if n > 1 else 0
    std = variance ** 0.5
    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "median": round(median, 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "n": n,
    }


def print_summary_table(model_name, block_stats, total_stats):
    """Print a formatted summary table to stdout."""
    print(f"\n{'=' * 70}")
    print(f"  {model_name} — Block-level Timing Summary (n={total_stats.get('n', 0)})")
    print(f"{'=' * 70}")
    print(f"  {'Block':<20} {'Mean':>8} {'Std':>8} {'Median':>8} {'Min':>8} {'Max':>8} {'%':>6}")
    print(f"  {'-' * 20} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 6}")

    total_mean = total_stats.get("mean", 1.0)
    for block_name, stats in block_stats.items():
        pct = (stats["mean"] / total_mean * 100) if total_mean > 0 else 0
        print(f"  {block_name:<20} {stats['mean']:>7.2f}s {stats['std']:>7.2f}s "
              f"{stats['median']:>7.2f}s {stats['min']:>7.2f}s {stats['max']:>7.2f}s {pct:>5.1f}%")

    print(f"  {'-' * 20} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 6}")
    print(f"  {'TOTAL':<20} {total_stats['mean']:>7.2f}s {total_stats['std']:>7.2f}s "
          f"{total_stats['median']:>7.2f}s {total_stats['min']:>7.2f}s {total_stats['max']:>7.2f}s {100.0:>5.1f}%")

    # Sanity check: blocks_sum vs total
    blocks_sum_mean = sum(s["mean"] for s in block_stats.values())
    overhead = total_mean - blocks_sum_mean
    print(f"\n  blocks_sum mean={blocks_sum_mean:.2f}s | total mean={total_mean:.2f}s | "
          f"overhead={overhead:.2f}s ({overhead / total_mean * 100:.1f}%)")
    print()


def main():
    parser = argparse.ArgumentParser(description="Analyze profiling results")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_root = cfg["output_root"]
    os.makedirs(output_root, exist_ok=True)

    all_per_sample_rows = []

    for model_cfg in cfg["models"]:
        model_name = model_cfg["name"]
        pred_root = model_cfg["predictions_root"]

        profiles = load_profiles(pred_root)
        if not profiles:
            print(f"No profiles found for {model_name} in {pred_root}")
            continue

        # Filter out failed samples
        valid = [p for p in profiles if p.get("status") != "failed"]
        failed_count = sum(1 for p in profiles if p.get("status") == "failed")
        print(f"{model_name}: {len(profiles)} profiles, "
              f"{failed_count} failed, {len(valid)} used for stats")

        if not valid:
            continue

        # Collect block names from first valid profile
        block_names = list(valid[0]["blocks"].keys())

        # Aggregate per-block values
        block_values = defaultdict(list)
        total_values = []
        for p in valid:
            for block_name in block_names:
                wall_sec = p["blocks"].get(block_name, {}).get("wall_sec", 0.0)
                block_values[block_name].append(wall_sec)
            total_values.append(p["total_sec"])

            # Per-sample row
            row = {
                "model": model_name,
                "object_id": p["object_id"],
                "category": p.get("category", ""),
                "total_sec": p["total_sec"],
                "blocks_sum_sec": p.get("blocks_sum_sec", 0.0),
                "peak_gpu_gb": p.get("peak_gpu_gb", 0.0),
            }
            for bn in block_names:
                row[f"block_{bn}_sec"] = p["blocks"].get(bn, {}).get("wall_sec", 0.0)
            all_per_sample_rows.append(row)

        # Compute stats
        block_stats = {bn: compute_stats(block_values[bn]) for bn in block_names}
        total_stats = compute_stats(total_values)

        # Print summary
        print_summary_table(model_name, block_stats, total_stats)

        # Write summary CSV
        summary_path = os.path.join(output_root, f"timing_summary_{model_name}.csv")
        with open(summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["block", "mean", "std", "median", "min", "max", "n", "pct_of_total"])
            for bn, stats in block_stats.items():
                pct = stats["mean"] / total_stats["mean"] * 100 if total_stats["mean"] > 0 else 0
                writer.writerow([bn, stats["mean"], stats["std"], stats["median"],
                                 stats["min"], stats["max"], stats["n"], round(pct, 1)])
            writer.writerow(["TOTAL", total_stats["mean"], total_stats["std"],
                             total_stats["median"], total_stats["min"], total_stats["max"],
                             total_stats["n"], 100.0])
        print(f"  Summary written to {summary_path}")

    # Write combined per-sample CSV
    if all_per_sample_rows:
        per_sample_path = os.path.join(output_root, "timing_per_sample.csv")
        fieldnames = list(all_per_sample_rows[0].keys())
        # Merge all fieldnames across models
        for row in all_per_sample_rows:
            for k in row:
                if k not in fieldnames:
                    fieldnames.append(k)
        with open(per_sample_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_per_sample_rows)
        print(f"Per-sample data written to {per_sample_path}")


if __name__ == "__main__":
    main()
