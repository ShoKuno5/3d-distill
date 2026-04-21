#!/usr/bin/env python3
"""Post-processing for E1: join diagnosis.csv with existing per_sample.csv,
classify samples, produce summary and figure.

Usage:
    python e1_postprocess.py \
      --diagnosis-csv results/.../e1_dmd1_volume_logit/diagnosis.csv \
      --per-sample-csv results/distill_methods/runs/20260421_cross_family/metrics/per_sample.csv \
      --model dmd1_1step \
      --out-dir  results/.../e1_dmd1_volume_logit
"""
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def load_per_sample(csv_path: Path, model_name: str, track: str = "A"):
    """Return dict: object_id -> {cd, category, ...} for given model+track."""
    out = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["model"] != model_name:
                continue
            if r.get("track") and r["track"] != track:
                continue
            try:
                cd = float(r["chamfer_distance"]) if r["chamfer_distance"] else np.nan
            except ValueError:
                cd = np.nan
            out[r["object_id"]] = {
                "cd": cd,
                "category": r["category"],
                "hausdorff": float(r["hausdorff"]) if r.get("hausdorff") else np.nan,
                "f_score_001": float(r["f_score_001"]) if r.get("f_score_001") else np.nan,
                "alignment_scale": float(r["alignment_scale"]) if r.get("alignment_scale") else np.nan,
            }
    return out


def cd_bucket(cd: float) -> str:
    if np.isnan(cd):
        return "nan"
    if cd < 5e-3:
        return "success"
    if cd < 5e-2:
        return "intermediate"
    return "failure"


def compute_latent_stats(latent_path: Path):
    """Per-token L2 norm distribution + aggregate stats."""
    if not latent_path.exists():
        return {}
    z = np.load(latent_path).astype(np.float64)  # (T, D)
    norms = np.linalg.norm(z, axis=-1)
    return {
        "latent_norm_mean": float(norms.mean()),
        "latent_norm_median": float(np.median(norms)),
        "latent_norm_p05": float(np.percentile(norms, 5)),
        "latent_norm_p95": float(np.percentile(norms, 95)),
        "latent_norm_std": float(norms.std()),
        "latent_mean_abs": float(np.abs(z).mean()),
        "latent_energy": float((z ** 2).sum()),
    }


def load_teacher_reference_norms(ref_dir: Path):
    """Return (mean, std) over all 10 step-50 teacher latents (reference distribution)."""
    norms_means = []
    for p in ref_dir.glob("*/latent_step_50.npy"):
        z = np.load(p).astype(np.float64)
        norms = np.linalg.norm(z, axis=-1)
        norms_means.append(float(norms.mean()))
    arr = np.asarray(norms_means)
    return {
        "teacher_mean_of_means": float(arr.mean()),
        "teacher_std_of_means": float(arr.std()),
        "teacher_min_of_means": float(arr.min()),
        "teacher_max_of_means": float(arr.max()),
        "n": len(norms_means),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diagnosis-csv", required=True, type=Path)
    ap.add_argument("--per-sample-csv", required=True, type=Path)
    ap.add_argument("--latents-dir", type=Path, default=None,
                    help="Dir containing <oid>.npy from diagnose_dmd1_manifold.py")
    ap.add_argument("--teacher-latents-dir", type=Path, default=None,
                    help="Dir with teacher reference step_50 latents for norm comparison")
    ap.add_argument("--model", default="dmd1_1step")
    ap.add_argument("--track", default="A")
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load diagnosis
    diag = {}
    with open(args.diagnosis_csv) as f:
        for r in csv.DictReader(f):
            if r["status"] != "success":
                diag[r["object_id"]] = r
                continue
            diag[r["object_id"]] = {
                "category": r["category"],
                "min": float(r["min"]) if r["min"] else np.nan,
                "max": float(r["max"]) if r["max"] else np.nan,
                "neg_pct": float(r["neg_pct"]) if r["neg_pct"] else np.nan,
                "std": float(r["std"]) if r["std"] else np.nan,
                "span": float(r["span"]) if r["span"] else np.nan,
                "zero_crossing": r["zero_crossing"] == "True",
                "mode_class": r["mode_class"],
                "status": r["status"],
            }

    # Load existing CD
    cd_data = load_per_sample(args.per_sample_csv, args.model, args.track)

    # Optional: teacher reference norms
    teacher_ref = None
    if args.teacher_latents_dir:
        teacher_ref = load_teacher_reference_norms(args.teacher_latents_dir)
        print(f"Teacher step-50 reference (n={teacher_ref['n']}): "
              f"mean_of_means={teacher_ref['teacher_mean_of_means']:.3f} "
              f"std={teacher_ref['teacher_std_of_means']:.3f} "
              f"range=[{teacher_ref['teacher_min_of_means']:.3f}, {teacher_ref['teacher_max_of_means']:.3f}]")

    # Join + classify
    joined = []
    for oid, d in diag.items():
        cd_entry = cd_data.get(oid)
        if cd_entry is None:
            cd = np.nan
            cat = d.get("category", "")
        else:
            cd = cd_entry["cd"]
            cat = cd_entry["category"]
        row = {
            "object_id": oid,
            "category": cat,
            "cd": cd,
            "cd_bucket": cd_bucket(cd),
            "mode_class": d.get("mode_class", "error"),
            "min": d.get("min", ""),
            "max": d.get("max", ""),
            "span": d.get("span", ""),
            "neg_pct": d.get("neg_pct", ""),
            "std": d.get("std", ""),
            "zero_crossing": d.get("zero_crossing", ""),
            "alignment_scale": cd_entry["alignment_scale"] if cd_entry else np.nan,
            "hausdorff": cd_entry["hausdorff"] if cd_entry else np.nan,
            "f_score_001": cd_entry["f_score_001"] if cd_entry else np.nan,
        }
        # Augment with latent stats
        if args.latents_dir:
            lat_stats = compute_latent_stats(args.latents_dir / f"{oid}.npy")
            row.update(lat_stats)
            if teacher_ref and "latent_norm_mean" in lat_stats:
                dev = (lat_stats["latent_norm_mean"] - teacher_ref["teacher_mean_of_means"]) / teacher_ref["teacher_std_of_means"]
                row["latent_norm_z"] = float(dev)
        joined.append(row)

    # Write joined CSV
    joined_path = args.out_dir / "joined.csv"
    with open(joined_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(joined[0].keys()))
        w.writeheader()
        for r in joined:
            w.writerow(r)
    print(f"Wrote {joined_path} ({len(joined)} rows)")

    # Summary: cross-tab mode_class × cd_bucket
    ct = defaultdict(lambda: Counter())
    for r in joined:
        ct[r["mode_class"]][r["cd_bucket"]] += 1

    print("\n=== Cross-tab: mode_class × cd_bucket ===")
    buckets = ["success", "intermediate", "failure", "nan"]
    hdr = f"{'mode_class':>12s}  " + "  ".join(f"{b:>12s}" for b in buckets) + "  total"
    print(hdr)
    for mc in sorted(ct.keys()):
        row = f"{mc:>12s}  " + "  ".join(f"{ct[mc][b]:>12d}" for b in buckets)
        total = sum(ct[mc].values())
        print(row + f"  {total:>5d}")

    # Mode breakdown within failures
    failures = [r for r in joined if r["cd_bucket"] == "failure"]
    print(f"\n=== Decision metric: Mode class in CD failures (CD >= 5e-2) ===")
    mc_in_fail = Counter(r["mode_class"] for r in failures)
    n_fail = len(failures)
    print(f"  n_failures: {n_fail}")
    if n_fail > 0:
        for mc, n in mc_in_fail.most_common():
            print(f"    {mc}: {n}  ({100*n/n_fail:.1f}%)")

    mode_a_rate_in_fail = mc_in_fail["A"] / n_fail if n_fail else 0
    print(f"\n  Mode A rate in failures: {mode_a_rate_in_fail*100:.1f}%")
    if mode_a_rate_in_fail > 0.8:
        verdict = "unified hypothesis supported — central figure 1 axis"
    elif mode_a_rate_in_fail < 0.2:
        verdict = "unified hypothesis broken — DMD1 failure is wrong-shape (Mode B), need 2 axes"
    else:
        verdict = "mixed — need to split figure"
    print(f"  Verdict: {verdict}")

    # Sanity: successes should be mode_class=B with zero_crossing
    successes = [r for r in joined if r["cd_bucket"] == "success"]
    mc_in_success = Counter(r["mode_class"] for r in successes)
    print(f"\n=== Sanity: Mode class in CD successes (CD < 5e-3, n={len(successes)}) ===")
    for mc, n in mc_in_success.most_common():
        pct = 100 * n / len(successes) if successes else 0
        print(f"    {mc}: {n}  ({pct:.1f}%)")

    # Additional analysis: latent norm distribution by CD bucket
    if args.latents_dir:
        print("\n=== Latent norm (per-token mean) by CD bucket ===")
        for bucket in ["success", "intermediate", "failure"]:
            vals = [float(r["latent_norm_mean"]) for r in joined
                    if r.get("cd_bucket") == bucket and "latent_norm_mean" in r]
            if vals:
                arr = np.asarray(vals)
                print(f"  {bucket:12s}: n={len(vals):3d}  mean={arr.mean():.3f}  std={arr.std():.3f}  "
                      f"range=[{arr.min():.3f}, {arr.max():.3f}]")
        # Scale-clamp saturation rate by bucket
        print("\n=== Alignment scale clamp saturation (|scale-0.5|<1e-3 OR |scale-2.0|<1e-3) by CD bucket ===")
        for bucket in ["success", "intermediate", "failure"]:
            clamp_rows = [r for r in joined if r.get("cd_bucket") == bucket]
            n = len(clamp_rows)
            sat = sum(1 for r in clamp_rows if isinstance(r.get("alignment_scale"), (int, float))
                      and (abs(r["alignment_scale"] - 0.5) < 1e-3 or abs(r["alignment_scale"] - 2.0) < 1e-3))
            pct = 100 * sat / n if n else 0
            print(f"  {bucket:12s}: {sat}/{n} ({pct:.1f}%) saturated")

    # Scatter plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        color_map = {"A": "#ef4444", "B": "#22c55e", "unclear": "#eab308", "error": "#64748b"}
        for mc in ["A", "B", "unclear", "error"]:
            xs, ys = [], []
            for r in joined:
                if r["mode_class"] != mc:
                    continue
                if r["span"] == "" or np.isnan(r["cd"]):
                    continue
                xs.append(float(r["span"]))
                ys.append(float(r["cd"]))
            if xs:
                ax.scatter(xs, ys, c=color_map[mc], alpha=0.65, s=30, label=f"Mode {mc} (n={len(xs)})", edgecolor="white", linewidth=0.5)
        ax.set_yscale("log")
        ax.set_xlabel("Volume-logit span (max - min) — higher = decoder produced non-degenerate SDF")
        ax.set_ylabel("Chamfer distance (log)")
        ax.axhline(5e-3, color="#22c55e", linestyle=":", alpha=0.5, label="CD=5e-3")
        ax.axhline(5e-2, color="#ef4444", linestyle=":", alpha=0.5, label="CD=5e-2")
        ax.axvline(0.05, color="black", linestyle=":", alpha=0.4)
        ax.set_title(f"E1: {args.model} — volume-logit span vs Chamfer (track {args.track})")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        scatter_path = args.out_dir / "scatter_span_vs_cd.png"
        fig.savefig(scatter_path, dpi=120)
        plt.close(fig)
        print(f"\nWrote {scatter_path}")

        # Second plot: CD bucket × mode_class bar chart
        fig, ax = plt.subplots(figsize=(8, 5))
        buckets_plot = ["success", "intermediate", "failure"]
        modes_plot = ["A", "B", "unclear"]
        x = np.arange(len(buckets_plot))
        width = 0.25
        for i, mc in enumerate(modes_plot):
            heights = [ct[mc][b] for b in buckets_plot]
            ax.bar(x + (i - 1) * width, heights, width, label=f"Mode {mc}", color=color_map[mc])
        ax.set_xticks(x)
        ax.set_xticklabels(buckets_plot)
        ax.set_ylabel("count")
        ax.set_title(f"E1: {args.model} — CD bucket × volume-logit mode")
        ax.legend()
        fig.tight_layout()
        bar_path = args.out_dir / "mode_vs_cd_bucket.png"
        fig.savefig(bar_path, dpi=120)
        plt.close(fig)
        print(f"Wrote {bar_path}")

        # Latent norm vs CD scatter (if available)
        if args.latents_dir and any("latent_norm_mean" in r for r in joined):
            fig, ax = plt.subplots(figsize=(8, 6))
            for bucket, color in [("success", "#22c55e"), ("intermediate", "#eab308"), ("failure", "#ef4444")]:
                xs = [float(r["latent_norm_mean"]) for r in joined
                      if r.get("cd_bucket") == bucket and "latent_norm_mean" in r]
                ys = [float(r["cd"]) for r in joined
                      if r.get("cd_bucket") == bucket and "latent_norm_mean" in r]
                if xs:
                    ax.scatter(xs, ys, c=color, alpha=0.7, s=30, label=f"{bucket} (n={len(xs)})",
                               edgecolor="white", linewidth=0.5)
            if teacher_ref:
                ax.axvspan(teacher_ref["teacher_min_of_means"],
                           teacher_ref["teacher_max_of_means"],
                           color="#38bdf8", alpha=0.1, label="teacher ref range")
                ax.axvline(teacher_ref["teacher_mean_of_means"], color="#38bdf8", linestyle=":", alpha=0.5)
            ax.set_yscale("log")
            ax.set_xlabel("per-token latent L2 norm (mean over 4096 tokens)")
            ax.set_ylabel("Chamfer distance (log)")
            ax.axhline(5e-3, color="#22c55e", linestyle=":", alpha=0.3)
            ax.axhline(5e-2, color="#ef4444", linestyle=":", alpha=0.3)
            ax.set_title(f"E1: {args.model} — latent norm vs CD (track {args.track})")
            ax.legend(loc="upper left", fontsize=9)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            norm_path = args.out_dir / "latent_norm_vs_cd.png"
            fig.savefig(norm_path, dpi=120)
            plt.close(fig)
            print(f"Wrote {norm_path}")

            # Alignment scale vs latent norm (to see correlation)
            fig, ax = plt.subplots(figsize=(8, 6))
            for bucket, color in [("success", "#22c55e"), ("intermediate", "#eab308"), ("failure", "#ef4444")]:
                xs = [float(r["latent_norm_mean"]) for r in joined
                      if r.get("cd_bucket") == bucket and "latent_norm_mean" in r
                      and isinstance(r.get("alignment_scale"), (int, float))]
                ys = [float(r["alignment_scale"]) for r in joined
                      if r.get("cd_bucket") == bucket and "latent_norm_mean" in r
                      and isinstance(r.get("alignment_scale"), (int, float))]
                if xs:
                    ax.scatter(xs, ys, c=color, alpha=0.7, s=30, label=f"{bucket} (n={len(xs)})",
                               edgecolor="white", linewidth=0.5)
            ax.axhline(0.5, color="black", linestyle="--", alpha=0.4, label="scale clamp floor")
            ax.axhline(2.0, color="black", linestyle="--", alpha=0.4, label="scale clamp ceiling")
            ax.axhline(1.0, color="gray", linestyle=":", alpha=0.3)
            ax.set_xlabel("per-token latent L2 norm mean")
            ax.set_ylabel("alignment_scale (ICP scale factor)")
            ax.set_title(f"E1: {args.model} — latent norm vs ICP scale")
            ax.legend(loc="best", fontsize=9)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            scale_norm_path = args.out_dir / "latent_norm_vs_alignment_scale.png"
            fig.savefig(scale_norm_path, dpi=120)
            plt.close(fig)
            print(f"Wrote {scale_norm_path}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Plot failed: {e}")


if __name__ == "__main__":
    main()
