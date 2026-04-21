#!/usr/bin/env python3
"""Aggregate fairness diagnostics across models in a cross-family benchmark run.

Reads per-sample CSVs from <RUN_DIR>/metrics and <RUN_DIR>/clamp_loose/metrics
and produces:
  - alignment_scale histogram per model
  - clamp hit rate (|scale - 0.5| < eps OR |scale - 2.0| < eps)
  - track_a vs track_b CD delta
  - scale clamp sensitivity: CD(loose) - CD(tight) per model
  - mesh quality stats summary (vertices, faces, components)
  - per-category breakdown (e.g. thin-object categories)

Output: <RUN_DIR>/diagnostics/{fairness_stats.csv, histograms.png,
  clamp_sensitivity.png, mesh_stats.png}
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


EPS = 1e-3


def load_per_sample(run_dir: Path, suffix: str = "") -> pd.DataFrame:
    """Load per-sample metrics from <run_dir>/<suffix>/metrics/per_sample.csv.

    The eval pipeline writes per_sample.csv per run (all models aggregated).
    Column of interest: object_id, category, model, track, chamfer_distance,
    alignment_scale, alignment_rot_idx, f_score_001, f_score_002, hausdorff.
    """
    metrics_dir = run_dir / suffix / "metrics" if suffix else run_dir / "metrics"
    csv_path = metrics_dir / "per_sample.csv"
    if not csv_path.exists():
        # fallback: per_sample_<model>.csv merged
        parts = []
        for p in metrics_dir.glob("per_sample_*.csv"):
            parts.append(pd.read_csv(p))
        if not parts:
            raise FileNotFoundError(f"No per_sample CSV found under {metrics_dir}")
        return pd.concat(parts, ignore_index=True)
    return pd.read_csv(csv_path)


def load_mesh_quality(run_dir: Path) -> pd.DataFrame:
    """Load mesh quality stats from <run_dir>/metrics/mesh_quality.csv."""
    p = run_dir / "metrics" / "mesh_quality.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def clamp_hit_rate(df: pd.DataFrame, s_min: float, s_max: float) -> pd.DataFrame:
    """Fraction of samples where alignment_scale == clamp endpoint per model."""
    out = []
    for model, grp in df[df["track"] == "A"].groupby("model"):
        sc = grp["alignment_scale"].to_numpy()
        hit_low = float(np.mean(np.abs(sc - s_min) < EPS))
        hit_high = float(np.mean(np.abs(sc - s_max) < EPS))
        out.append({
            "model": model,
            "n_samples": len(sc),
            "scale_mean": float(np.mean(sc)),
            "scale_median": float(np.median(sc)),
            "scale_std": float(np.std(sc)),
            "clamp_hit_low_pct": 100 * hit_low,
            "clamp_hit_high_pct": 100 * hit_high,
            "clamp_hit_any_pct": 100 * (hit_low + hit_high),
        })
    return pd.DataFrame(out).sort_values("model")


def track_ab_delta(df: pd.DataFrame) -> pd.DataFrame:
    """Mean CD for Track A vs Track B per model; delta = CD_B - CD_A (positive = ICP rotation helped)."""
    out = []
    for model in sorted(df["model"].unique()):
        mdf = df[df["model"] == model]
        cd_a = mdf[mdf["track"] == "A"]["chamfer_distance"].mean()
        cd_b = mdf[mdf["track"] == "B"]["chamfer_distance"].mean()
        out.append({
            "model": model,
            "cd_track_a_mean": cd_a,
            "cd_track_b_mean": cd_b,
            "cd_icp_gain": (cd_b - cd_a) if pd.notna(cd_b) else None,
        })
    return pd.DataFrame(out)


def clamp_sensitivity(df_tight: pd.DataFrame, df_loose: pd.DataFrame) -> pd.DataFrame:
    """Scale clamp sensitivity: CD(loose) - CD(tight) per model (Track A only)."""
    tight = df_tight[df_tight["track"] == "A"].groupby("model")["chamfer_distance"].mean()
    loose = df_loose[df_loose["track"] == "A"].groupby("model")["chamfer_distance"].mean()
    out = []
    for model in sorted(set(tight.index) | set(loose.index)):
        out.append({
            "model": model,
            "cd_clamp_tight": tight.get(model, None),
            "cd_clamp_loose": loose.get(model, None),
            "cd_delta": (loose.get(model, np.nan) - tight.get(model, np.nan)),
        })
    return pd.DataFrame(out)


def plot_scale_histograms(df: pd.DataFrame, out_path: Path, s_min: float, s_max: float):
    models = sorted(df["model"].unique())
    n = len(models)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 2.8))
    axes = np.array(axes).reshape(-1)
    for ax, model in zip(axes, models):
        sc = df[(df["model"] == model) & (df["track"] == "A")]["alignment_scale"].to_numpy()
        ax.hist(sc, bins=30, range=(s_min * 0.9, s_max * 1.05), color="steelblue", edgecolor="black")
        ax.axvline(s_min, color="red", linestyle="--", lw=1, alpha=0.7)
        ax.axvline(s_max, color="red", linestyle="--", lw=1, alpha=0.7)
        ax.axvline(1.0, color="green", linestyle=":", lw=1, alpha=0.7)
        ax.set_title(f"{model} (n={len(sc)})", fontsize=9)
        ax.set_xlabel("alignment_scale")
        ax.set_ylabel("count")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Track A alignment_scale distribution per model", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_clamp_sensitivity(df_sens: pd.DataFrame, out_path: Path):
    if df_sens.empty or df_sens["cd_delta"].isna().all():
        return
    df_sens = df_sens.sort_values("cd_delta", ascending=False)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(df_sens))))
    ax.barh(df_sens["model"], df_sens["cd_delta"], color="tomato")
    ax.set_xlabel("CD(clamp=[0.3, 3.0]) − CD(clamp=[0.5, 2.0])")
    ax.set_title("Scale clamp sensitivity per model\n(positive → tight clamp was masking error)")
    ax.axvline(0, color="black", lw=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_mesh_stats(mq: pd.DataFrame, out_path: Path):
    if mq.empty:
        return
    stats = ["vertices", "faces", "connected_components"]
    cols = [c for c in stats if c in mq.columns]
    if not cols:
        return
    fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 4))
    axes = np.atleast_1d(axes)
    for ax, col in zip(axes, cols):
        models = sorted(mq["model"].unique())
        data = [mq[mq["model"] == m][col].dropna().to_numpy() for m in models]
        ax.boxplot(data, labels=models, showfliers=False)
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=45)
        ax.set_yscale("log" if col != "connected_components" else "linear")
    fig.suptitle("Mesh quality stats per model (post-cleaning)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="runs/<RUN_ID>/ root")
    ap.add_argument("--scale-clamp-tight", nargs=2, type=float, default=[0.5, 2.0])
    ap.add_argument("--clamp-loose-subdir", default="clamp_loose",
                    help="Subdir under run-dir containing sensitivity-run metrics")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    out_dir = run_dir / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading default metrics from {run_dir}/metrics ...")
    df_default = load_per_sample(run_dir)
    print(f"  {len(df_default)} rows, {df_default['model'].nunique()} models")

    df_loose = pd.DataFrame()
    sens_path = run_dir / args.clamp_loose_subdir
    if sens_path.exists():
        try:
            df_loose = load_per_sample(sens_path)
            print(f"  sensitivity: {len(df_loose)} rows")
        except FileNotFoundError as e:
            print(f"  no sensitivity metrics: {e}")

    mq = load_mesh_quality(run_dir)
    print(f"  mesh_quality rows: {len(mq)}")

    s_min, s_max = args.scale_clamp_tight
    stats_hit = clamp_hit_rate(df_default, s_min, s_max)
    stats_ab = track_ab_delta(df_default)

    merged = stats_hit.merge(stats_ab, on="model", how="outer")
    if not df_loose.empty:
        stats_sens = clamp_sensitivity(df_default, df_loose)
        merged = merged.merge(stats_sens, on="model", how="outer")

    merged.to_csv(out_dir / "fairness_stats.csv", index=False)
    print(f"Wrote {out_dir / 'fairness_stats.csv'}")
    print(merged.to_string(index=False))

    print("\nPlotting ...")
    plot_scale_histograms(df_default, out_dir / "alignment_scale_hist.png", s_min, s_max)
    if not df_loose.empty:
        plot_clamp_sensitivity(
            clamp_sensitivity(df_default, df_loose),
            out_dir / "clamp_sensitivity.png",
        )
    if not mq.empty:
        plot_mesh_stats(mq, out_dir / "mesh_stats.png")

    print(f"\nDone. Diagnostics at {out_dir}")


if __name__ == "__main__":
    main()
