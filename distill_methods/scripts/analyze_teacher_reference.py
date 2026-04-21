#!/usr/bin/env python3
"""Aggregate teacher reference statistics from N=105 saved latents.

Input: teacher_ref_n105/latents/<oid>.npy produced by
       diagnose_dmd1_manifold.py --model-name teacher_50step

Outputs:
  teacher_ref_n105/stats.json        — aggregate + per-category statistics
  teacher_ref_n105/per_sample_norm.csv
  teacher_ref_n105/category_boxplot.png
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

N10_MEAN = 7.832
N10_STD = 0.115


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir", required=True, type=Path,
                    help="teacher_ref_n105 directory")
    ap.add_argument("--manifest", required=True, type=Path,
                    help="manifest_test.csv for category lookup")
    ap.add_argument("--diagnosis-csv", type=Path, default=None,
                    help="diagnosis.csv from teacher inference (for sanity cross-check)")
    args = ap.parse_args()

    # Load category map from manifest
    cat_map = {}
    with open(args.manifest) as f:
        for r in csv.DictReader(f):
            cat_map[r["object_id"]] = r.get("category", "unknown")

    # Walk latents
    latent_dir = args.ref_dir / "latents"
    rows = []
    per_cat = defaultdict(list)
    for p in sorted(latent_dir.glob("*.npy")):
        oid = p.stem
        z = np.load(p).astype(np.float64)  # (T, D)
        norm_mean = float(np.linalg.norm(z, axis=-1).mean())
        cat = cat_map.get(oid, "unknown")
        rows.append({"object_id": oid, "category": cat, "latent_norm_mean": norm_mean})
        per_cat[cat].append(norm_mean)

    n = len(rows)
    all_norms = np.asarray([r["latent_norm_mean"] for r in rows])

    overall = {
        "mean": float(all_norms.mean()),
        "std": float(all_norms.std(ddof=1)),
        "median": float(np.median(all_norms)),
        "min": float(all_norms.min()),
        "max": float(all_norms.max()),
        "p05": float(np.percentile(all_norms, 5)),
        "p95": float(np.percentile(all_norms, 95)),
    }

    # Per-category
    per_cat_stats = {}
    for c, vals in sorted(per_cat.items()):
        arr = np.asarray(vals)
        per_cat_stats[c] = {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "delta_from_overall": float(arr.mean() - overall["mean"]),
        }

    # Per-sample norm deviations for outlier detection
    z_scores = (all_norms - overall["mean"]) / overall["std"]
    outliers = [
        (rows[i]["object_id"], rows[i]["category"], float(all_norms[i]), float(z_scores[i]))
        for i in range(n) if abs(z_scores[i]) > 3.0
    ]
    outliers.sort(key=lambda x: abs(x[3]), reverse=True)

    stats = {
        "n_samples": n,
        "seed": 42,
        "latent_shape": list(np.load(next(latent_dir.glob("*.npy"))).shape),
        "overall": overall,
        "comparison_to_n10": {
            "n10_mean": N10_MEAN,
            "n10_std": N10_STD,
            "delta_mean": overall["mean"] - N10_MEAN,
            "delta_std": overall["std"] - N10_STD,
            "n10_within_se": abs(overall["mean"] - N10_MEAN) < 0.04,
        },
        "per_category": per_cat_stats,
        "outliers_abs_z_gt_3": [
            {"object_id": o, "category": c, "norm_mean": n_, "z_score": z_}
            for o, c, n_, z_ in outliers
        ],
    }

    (args.ref_dir / "stats.json").write_text(json.dumps(stats, indent=2))

    # per_sample_norm.csv
    with open(args.ref_dir / "per_sample_norm.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["object_id", "category", "latent_norm_mean", "z_score"])
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({**r, "z_score": float(z_scores[i])})

    # Console report
    print(f"=== Teacher reference (N={n}) ===")
    print(f"Overall norm: mean={overall['mean']:.4f}  std={overall['std']:.4f}  "
          f"range=[{overall['min']:.4f}, {overall['max']:.4f}]")
    print(f"              median={overall['median']:.4f}  p05={overall['p05']:.4f}  p95={overall['p95']:.4f}")
    print(f"")
    print(f"vs N=10 ({N10_MEAN:.3f} ± {N10_STD:.3f}):")
    dm = stats["comparison_to_n10"]["delta_mean"]
    ds = stats["comparison_to_n10"]["delta_std"]
    within = "YES" if stats["comparison_to_n10"]["n10_within_se"] else "NO"
    print(f"  delta_mean = {dm:+.4f}  (within ±0.04: {within})")
    print(f"  delta_std  = {ds:+.4f}")
    print(f"")
    print(f"=== Per-category (sorted by |delta from overall|) ===")
    sorted_cats = sorted(per_cat_stats.items(), key=lambda kv: abs(kv[1]["delta_from_overall"]), reverse=True)
    for c, s in sorted_cats[:20]:
        print(f"  {c:20s}  n={s['n']:3d}  mean={s['mean']:.4f}  std={s['std']:.4f}  delta={s['delta_from_overall']:+.4f}")
    if len(sorted_cats) > 20:
        print(f"  ... ({len(sorted_cats) - 20} more categories)")

    # Universality summary
    cat_means = np.asarray([s["mean"] for s in per_cat_stats.values()])
    print(f"")
    print(f"=== Category-invariance check ===")
    print(f"  n_categories: {len(per_cat_stats)}")
    print(f"  category-mean spread: std={cat_means.std(ddof=1):.4f}  range=[{cat_means.min():.4f}, {cat_means.max():.4f}]")
    print(f"  max |delta from overall|: {max(abs(s['delta_from_overall']) for s in per_cat_stats.values()):.4f}")
    within_01 = sum(1 for s in per_cat_stats.values() if abs(s["delta_from_overall"]) <= 0.1)
    print(f"  categories within ±0.1 of overall: {within_01}/{len(per_cat_stats)}")

    if outliers:
        print(f"")
        print(f"=== Outliers (|z| > 3) ===")
        for oid, cat, val, z in outliers:
            print(f"  {oid:25s}  cat={cat:15s}  norm={val:.4f}  z={z:+.2f}")

    # Optional diagnosis.csv cross-check
    if args.diagnosis_csv and args.diagnosis_csv.exists():
        with open(args.diagnosis_csv) as f:
            diag = list(csv.DictReader(f))
        modes = defaultdict(int)
        for r in diag:
            modes[r["mode_class"]] += 1
        print(f"")
        print(f"=== diagnosis.csv mode breakdown (sanity) ===")
        for m, c in sorted(modes.items()):
            print(f"  {m}: {c}")

    # Category box plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cats_sorted = sorted(per_cat.keys(), key=lambda c: per_cat_stats[c]["mean"])
        data = [per_cat[c] for c in cats_sorted]

        fig, ax = plt.subplots(figsize=(max(10, len(cats_sorted) * 0.18), 5.5))
        ax.set_facecolor("#0f172a")
        fig.patch.set_facecolor("#0f172a")
        bp = ax.boxplot(data, positions=range(len(cats_sorted)), widths=0.65,
                        patch_artist=True, showmeans=True,
                        flierprops={"marker": "o", "markersize": 3, "markerfacecolor": "#ef4444", "markeredgecolor": "#ef4444"})
        for patch in bp["boxes"]:
            patch.set_facecolor("#1e293b")
            patch.set_edgecolor("#38bdf8")
        for med in bp["medians"]:
            med.set_color("#eab308")
        for w in bp["whiskers"]:
            w.set_color("#94a3b8")
        for c in bp["caps"]:
            c.set_color("#94a3b8")
        for m in bp["means"]:
            m.set_marker("D")
            m.set_markerfacecolor("#22c55e")
            m.set_markeredgecolor("#22c55e")
            m.set_markersize(5)

        ax.axhline(overall["mean"], color="#38bdf8", linestyle="--", alpha=0.6, label=f"overall mean = {overall['mean']:.3f}")
        ax.axhspan(overall["mean"] - overall["std"], overall["mean"] + overall["std"],
                   color="#38bdf8", alpha=0.08, label="± 1σ")
        ax.axhline(N10_MEAN, color="#a78bfa", linestyle=":", alpha=0.7, label=f"N=10 reference = {N10_MEAN:.3f}")
        ax.set_xticks(range(len(cats_sorted)))
        ax.set_xticklabels(cats_sorted, rotation=75, ha="right", fontsize=7.5, color="#94a3b8")
        ax.tick_params(colors="#94a3b8")
        ax.set_ylabel("per-token latent L2 norm (mean)", color="#e2e8f0")
        ax.set_title(f"Teacher step-50 latent norm, N={n} samples across {len(cats_sorted)} categories",
                     color="#e2e8f0", fontsize=11)
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.grid(True, alpha=0.15, axis="y", color="#64748b")
        ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=9, loc="lower right")
        plt.tight_layout()
        plt.savefig(args.ref_dir / "category_boxplot.png", dpi=140, facecolor="#0f172a", bbox_inches="tight")
        plt.close()
        print(f"")
        print(f"Wrote {args.ref_dir / 'category_boxplot.png'}")
    except Exception as e:
        print(f"Boxplot failed: {e}")

    print(f"Wrote {args.ref_dir / 'stats.json'}")
    print(f"Wrote {args.ref_dir / 'per_sample_norm.csv'}")


if __name__ == "__main__":
    main()
