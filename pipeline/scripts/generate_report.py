#!/usr/bin/env python3
"""Generate evaluation report from computed metrics.

Usage:
    python pipeline/scripts/generate_report.py \
        --config experiments/toys4k_baseline/config.yaml
"""

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))


def load_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_root = cfg["output_root"]
    metrics_dir = os.path.join(output_root, "metrics")

    per_sample = load_csv(os.path.join(metrics_dir, "per_sample.csv"))
    summary = load_csv(os.path.join(metrics_dir, "summary.csv"))
    mesh_quality = load_csv(os.path.join(metrics_dir, "mesh_quality.csv"))
    timing = load_csv(os.path.join(metrics_dir, "timing.csv"))
    failures = load_csv(os.path.join(metrics_dir, "failures.csv"))

    # Sample IDs used
    ids_path = os.path.join(output_root, "cache", "sample_ids.txt")
    sample_ids = []
    if os.path.exists(ids_path):
        with open(ids_path) as f:
            sample_ids = [l.strip() for l in f if l.strip()]

    model_names = [m["name"] for m in cfg["models"]]

    report = []
    report.append("# Toys4k Aligned Geometry Evaluation Report")
    report.append("")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    # --- Experiment overview ---
    report.append("## Experiment Overview")
    report.append("")
    n_models = len(model_names)
    report.append(f"- **Purpose**: Fair geometry comparison of {n_models} single-image 3D reconstruction models")
    report.append(f"- **Dataset**: Toys4k (subset)")
    report.append(f"- **Models**: {', '.join(model_names)}")
    report.append(f"- **Samples evaluated**: {len(sample_ids)}")
    report.append(f"- **Seeds**: {cfg.get('seeds', [42])}")
    report.append("")

    # --- Model versions ---
    report.append("## Model Versions")
    report.append("")
    report.append("| Model | Version | Status |")
    report.append("|-------|---------|--------|")
    for m in cfg["models"]:
        ver = m.get("version", "?")
        status = m.get("status", "unknown")
        report.append(f"| {m['name']} | {ver} | {status} |")
    report.append("")

    # --- Subset ---
    report.append("## Subset Definition")
    report.append("")
    report.append(f"Manifest: `{cfg['dataset']['manifest']}`")
    report.append("")
    report.append("| # | Object ID | Category |")
    report.append("|---|-----------|----------|")
    for i, sid in enumerate(sample_ids):
        cat = ""
        for r in per_sample:
            if r["object_id"] == sid:
                cat = r["category"]
                break
        report.append(f"| {i+1} | {sid} | {cat} |")
    report.append("")

    # --- Protocol ---
    report.append("## Evaluation Protocol")
    report.append("")
    report.append("### Input Preparation")
    report.append("- All models receive the same input image per sample (from Toys4k blend renders)")
    report.append("- No model-specific pre-processing is applied to inputs")
    report.append("")

    report.append("### GT Canonicalization")
    gt_method = cfg["normalization"]["gt_method"]
    report.append(f"- Method: `{gt_method}`")
    report.append("- Center: bbox center translated to origin")
    report.append("- Scale: isotropic, so that max point norm = 1 (unit sphere)")
    report.append("")

    report.append("### Prediction Mesh Cleaning")
    report.append("Applied identically to all models:")
    report.append(f"- NaN removal: {cfg['cleaning']['remove_nan']}")
    report.append(f"- Inf removal: {cfg['cleaning']['remove_inf']}")
    report.append(f"- Degenerate face removal: {cfg['cleaning']['remove_degenerate_faces']}")
    report.append(f"- Unreferenced vertex removal: {cfg['cleaning']['remove_unreferenced_vertices']}")
    report.append(f"- Tiny component removal: {cfg['cleaning']['remove_tiny_components']} (threshold: {cfg['cleaning']['tiny_component_threshold']})")
    report.append("- **No smoothing, no manual rotation, no model-specific post-processing**")
    report.append("")

    report.append("### Point Cloud Sampling")
    report.append(f"- Alignment: {cfg['sampling']['alignment_points']:,} points (area-proportional)")
    report.append(f"- Evaluation: {cfg['sampling']['eval_points']:,} points (area-proportional)")
    report.append("")

    report.append("### Track A: Similarity ICP Aligned")
    report.append("- Purpose: Main leaderboard — best achievable geometric accuracy")
    report.append("- Allowed: rotation + translation + uniform scale")
    report.append("- **Forbidden: reflection, non-uniform scale, manual adjustment**")
    report.append(f"- Initial rotations: {cfg['alignment']['track_a']['initial_rotations']} (cube symmetry group)")
    report.append(f"- ICP max iterations: {cfg['alignment']['track_a']['icp_max_iterations']}")
    report.append("- Alignment method: Umeyama similarity ICP (multi-start)")
    report.append("")

    report.append("### Track B: Pose-Aware (Scale Only)")
    report.append("- Purpose: Assess canonical pose stability")
    report.append("- Allowed: center + uniform scale normalization only")
    report.append("- **No rotation alignment applied**")
    report.append("")

    report.append("### Metrics")
    report.append("- **Chamfer Distance (CD)**: bilateral mean of squared L2 distances")
    report.append("  - `CD = mean(d²(pred→GT)) + mean(d²(GT→pred))`")
    report.append("- **F-score @ τ=0.01**: harmonic mean of precision/recall at distance threshold 0.01")
    report.append("- **F-score @ τ=0.02**: same at threshold 0.02")
    report.append("- **Hausdorff Distance**: max of directed Hausdorff in both directions")
    report.append("- Bootstrap 95% CI computed with 1000 resamples")
    report.append("")

    # --- Track A Results ---
    report.append("## Results — Track A (Similarity ICP Aligned)")
    report.append("")
    track_a_summary = [r for r in summary if r.get("track") == "A"]
    if track_a_summary:
        report.append("| Model | CD (×10³) | Hausdorff | F@1% | F@2% | N |")
        report.append("|-------|-----------|-----------|------|------|---|")
        for r in track_a_summary:
            cd = float(r.get("chamfer_distance_mean", "nan")) * 1000
            hd = float(r.get("hausdorff_mean", "nan"))
            f1 = float(r.get("f_score_001_mean", "nan"))
            f2 = float(r.get("f_score_002_mean", "nan"))
            n = r.get("n_samples", "?")
            report.append(f"| {r['model']} | {cd:.3f} | {hd:.4f} | {f1:.4f} | {f2:.4f} | {n} |")
        report.append("")

        # CI table
        has_ci = any("chamfer_distance_ci95_lo" in r for r in track_a_summary)
        if has_ci:
            report.append("### 95% Bootstrap Confidence Intervals")
            report.append("")
            report.append("| Model | CD (×10³) [CI] | F@1% [CI] | F@2% [CI] |")
            report.append("|-------|----------------|-----------|-----------|")
            for r in track_a_summary:
                cd_lo = float(r.get("chamfer_distance_ci95_lo") or "nan") * 1000
                cd_hi = float(r.get("chamfer_distance_ci95_hi") or "nan") * 1000
                f1_lo = float(r.get("f_score_001_ci95_lo") or "nan")
                f1_hi = float(r.get("f_score_001_ci95_hi") or "nan")
                f2_lo = float(r.get("f_score_002_ci95_lo") or "nan")
                f2_hi = float(r.get("f_score_002_ci95_hi") or "nan")
                report.append(
                    f"| {r['model']} | [{cd_lo:.3f}, {cd_hi:.3f}] | "
                    f"[{f1_lo:.4f}, {f1_hi:.4f}] | [{f2_lo:.4f}, {f2_hi:.4f}] |"
                )
            report.append("")

    # --- Per-sample detail ---
    report.append("### Per-Sample Detail (Track A)")
    report.append("")
    track_a_rows = [r for r in per_sample if r.get("track") == "A"]
    if track_a_rows:
        report.append("| Object | Model | CD (×10³) | HD | F@1% | F@2% | Scale |")
        report.append("|--------|-------|-----------|----|------|------|-------|")
        for r in sorted(track_a_rows, key=lambda x: (x["object_id"], x["model"])):
            cd = float(r["chamfer_distance"]) * 1000
            hd = float(r["hausdorff"])
            f1 = float(r["f_score_001"])
            f2 = float(r["f_score_002"])
            sc = float(r["alignment_scale"])
            report.append(
                f"| {r['object_id']} | {r['model']} | {cd:.3f} | {hd:.4f} | "
                f"{f1:.4f} | {f2:.4f} | {sc:.3f} |"
            )
        report.append("")

    # --- Track B ---
    track_b_summary = [r for r in summary if r.get("track") == "B"]
    if track_b_summary:
        report.append("## Results — Track B (Scale Only, No Rotation)")
        report.append("")
        report.append("| Model | CD (×10³) | Hausdorff | F@1% | F@2% | N |")
        report.append("|-------|-----------|-----------|------|------|---|")
        for r in track_b_summary:
            cd = float(r.get("chamfer_distance_mean", "nan")) * 1000
            hd = float(r.get("hausdorff_mean", "nan"))
            f1 = float(r.get("f_score_001_mean", "nan"))
            f2 = float(r.get("f_score_002_mean", "nan"))
            n = r.get("n_samples", "?")
            report.append(f"| {r['model']} | {cd:.3f} | {hd:.4f} | {f1:.4f} | {f2:.4f} | {n} |")
        report.append("")

    # --- Mesh quality ---
    if mesh_quality:
        report.append("## Mesh Quality Statistics")
        report.append("")
        report.append("| Model | Avg Verts | Avg Faces | Avg Components | Watertight % | Manifold % |")
        report.append("|-------|-----------|-----------|----------------|-------------|-----------|")
        for model in model_names:
            mrows = [r for r in mesh_quality if r["model"] == model]
            if not mrows:
                continue
            avg_v = np.mean([int(r["vertex_count"]) for r in mrows])
            avg_f = np.mean([int(r["face_count"]) for r in mrows])
            avg_c = np.mean([int(r["connected_components"]) for r in mrows])
            wt = 100 * np.mean([r["is_watertight"] == "True" for r in mrows])
            mf = 100 * np.mean([r["is_manifold"] == "True" for r in mrows])
            report.append(f"| {model} | {avg_v:.0f} | {avg_f:.0f} | {avg_c:.1f} | {wt:.0f}% | {mf:.0f}% |")
        report.append("")

    # --- Timing ---
    if timing:
        report.append("## Evaluation Timing")
        report.append("")
        report.append("| Model | Avg Total (s) | Avg Align (s) | Avg Metrics (s) |")
        report.append("|-------|---------------|---------------|-----------------|")
        for model in model_names:
            trows = [r for r in timing if r["model"] == model]
            if not trows:
                continue
            avg_total = np.mean([float(r["total_sec"]) for r in trows])
            avg_align = np.mean([float(r["align_sec"]) for r in trows])
            avg_met = np.mean([float(r["metrics_sec"]) for r in trows])
            report.append(f"| {model} | {avg_total:.2f} | {avg_align:.2f} | {avg_met:.2f} |")
        report.append("")

    # --- Failures ---
    if failures:
        report.append("## Failures")
        report.append("")
        report.append("| Object | Model | Error |")
        report.append("|--------|-------|-------|")
        for f in failures:
            report.append(f"| {f['object_id']} | {f['model']} | {f['error']} |")
        report.append("")

    # --- Fairness discussion ---
    report.append("## Fairness Assessment")
    report.append("")
    report.append("### What is fair")
    report.append("- Same input image for all models per sample")
    report.append("- Same mesh cleaning pipeline (no model-specific post-processing)")
    report.append("- Same point sampling (area-proportional, same seed)")
    report.append("- Same normalization (unit sphere)")
    report.append("- Same alignment (24-rotation multi-start similarity ICP)")
    report.append("- Same metrics computation")
    report.append("- Raw meshes used (before any model-specific simplification)")
    report.append("")
    report.append("### Potential fairness concerns")
    report.append("- Models may have different internal pre-processing (background removal, etc.)")
    report.append("- Models output meshes at different resolutions (vertex/face counts vary)")
    report.append("- Point sampling density relative to mesh resolution may differ")
    report.append("- Toys4k renders may favor some models over others (background, resolution)")
    report.append("- Seed=42 is used but models may have different determinism guarantees")
    report.append("- TRELLIS v1 vs v2 use different architectures (SLAT vs O-Voxel) — output characteristics may differ fundamentally")
    report.append("- Hunyuan3D 2.0 vs 2.1 use different model weights and may have different default inference parameters")
    report.append("- TRELLIS.2 pipeline_type (512 vs 1024_cascade) affects output resolution — document which was used")
    report.append("")

    # --- Next steps ---
    report.append("## Next Steps for Full Evaluation")
    report.append("")
    report.append("1. Expand to full Toys4k dataset (~4000 objects)")
    report.append("2. Run with multiple seeds (e.g., [0, 1, 2])")
    report.append("3. Add inference timing comparison")
    report.append("4. Add qualitative visualization (renders)")
    report.append("5. Investigate failure patterns per model")
    report.append("6. Consider adding Volume IoU and Normal Consistency metrics")
    report.append("")

    # Write report
    report_path = os.path.join(output_root, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report))
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
