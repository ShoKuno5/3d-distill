#!/usr/bin/env python3
"""Phase III direction drift analysis (E2).

For each sample with teacher ODE trajectory latents, compute per-token
cosine similarity and norm ratio between step pairs in Phase III (t=0.7..1.0).
Decides whether Phase III is:
  (a) uniform amplitude scaling   — cos~1, gini<0.1
  (b) selective amplification     — cos~1, gini>0.3
  (c) direction drift             — cos<0.95

Usage:
    python analyze_phase3_drift.py \
      --latent-dir results/distill_methods/intermediate_decode_cfg5 \
      --out-dir    results/distill_methods/runs/20260421_manifold_diagnosis/e2_phase3_drift
"""
import argparse
import csv
import os
from pathlib import Path

import numpy as np

# Order by shape complexity (matches report §3)
SAMPLE_ORDER = [
    ("ball_002", "simple"),
    ("apple_007", "simple"),
    ("cup_056", "medium"),
    ("shoe_028", "medium"),
    ("chair_168", "medium"),
    ("guitar_003", "complex"),
    ("airplane_007", "complex"),
    ("robot_016", "complex"),
    ("dinosaur_069", "complex"),
    ("helicopter_015", "complex"),
]

# Step pairs to analyse. Full trajectory coverage to locate where direction
# actually changes, not just Phase III.
STEP_PAIRS = [
    (5, 10), (10, 15), (15, 20), (20, 25), (25, 30),  # Phase I → II
    (30, 35), (35, 40), (40, 45), (45, 50),           # Phase II → III → tail
    (20, 50), (25, 50), (30, 50),                     # wide-span references
    (35, 50), (40, 50),                               # Phase III aggregates (kept)
]


def load_latent(path: Path):
    if not path.exists():
        return None
    return np.load(path).astype(np.float64)  # (4096, 64)


def gini(x: np.ndarray) -> float:
    """Gini coefficient over non-negative values."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or np.all(x == 0):
        return 0.0
    x = np.sort(np.clip(x, 0, None))
    n = x.size
    cum = np.cumsum(x)
    return (n + 1 - 2 * cum.sum() / cum[-1]) / n


def per_token_cos(z_a: np.ndarray, z_b: np.ndarray) -> np.ndarray:
    """Per-token cosine similarity between two (T, D) latents."""
    na = np.linalg.norm(z_a, axis=-1) + 1e-12
    nb = np.linalg.norm(z_b, axis=-1) + 1e-12
    return (z_a * z_b).sum(axis=-1) / (na * nb)


def per_token_norm_ratio(z_a: np.ndarray, z_b: np.ndarray) -> np.ndarray:
    na = np.linalg.norm(z_a, axis=-1) + 1e-12
    nb = np.linalg.norm(z_b, axis=-1) + 1e-12
    return nb / na


def classify(cos_mean: float, norm_gini: float) -> str:
    if cos_mean < 0.95:
        return "direction_drift"
    if norm_gini < 0.1:
        return "uniform_amplitude"
    if norm_gini > 0.3:
        return "selective_amplification"
    return "mixed"


def analyze_sample(oid: str, latent_dir: Path):
    latents = {}
    for step in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
        p = latent_dir / oid / f"latent_step_{step:02d}.npy"
        z = load_latent(p)
        if z is not None:
            latents[step] = z
    rows = []
    arr_dump = {}
    for s_ref, s_end in STEP_PAIRS:
        if s_ref not in latents or s_end not in latents:
            continue
        cos = per_token_cos(latents[s_ref], latents[s_end])
        ratio = per_token_norm_ratio(latents[s_ref], latents[s_end])
        # Token groups by norm-change: top 10% vs bottom 10%
        idx_sorted = np.argsort(ratio)
        n_tokens = len(ratio)
        q10 = max(1, n_tokens // 10)
        bottom = idx_sorted[:q10]
        top = idx_sorted[-q10:]
        rows.append({
            "object_id": oid,
            "step_ref": s_ref,
            "step_end": s_end,
            "cos_mean": float(cos.mean()),
            "cos_median": float(np.median(cos)),
            "cos_p05": float(np.percentile(cos, 5)),
            "cos_p95": float(np.percentile(cos, 95)),
            "cos_min": float(cos.min()),
            "ratio_mean": float(ratio.mean()),
            "ratio_median": float(np.median(ratio)),
            "ratio_std": float(ratio.std()),
            "ratio_gini": float(gini(ratio)),
            "ratio_p05": float(np.percentile(ratio, 5)),
            "ratio_p95": float(np.percentile(ratio, 95)),
            "cos_top10_mean": float(cos[top].mean()),
            "cos_bot10_mean": float(cos[bottom].mean()),
            "ratio_top10_mean": float(ratio[top].mean()),
            "ratio_bot10_mean": float(ratio[bottom].mean()),
            "n_tokens": int(n_tokens),
            "classification": classify(float(cos.mean()), float(gini(ratio))),
        })
        arr_dump[f"{oid}_{s_ref}_{s_end}_cos"] = cos.astype(np.float32)
        arr_dump[f"{oid}_{s_ref}_{s_end}_ratio"] = ratio.astype(np.float32)
    return rows, arr_dump


def plot_histograms(all_arrays: dict, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Grid: rows = samples (10), cols = 2 (cos hist, ratio hist) for the 35→50 pair
    fig, axes = plt.subplots(len(SAMPLE_ORDER), 2, figsize=(10, 2 * len(SAMPLE_ORDER)), sharey=False)
    for row, (oid, tier) in enumerate(SAMPLE_ORDER):
        cos_key = f"{oid}_35_50_cos"
        rat_key = f"{oid}_35_50_ratio"
        ax_cos = axes[row, 0]
        ax_rat = axes[row, 1]
        if cos_key in all_arrays:
            ax_cos.hist(all_arrays[cos_key], bins=60, range=(0.5, 1.01), color="#38bdf8", alpha=0.85)
            ax_cos.axvline(0.98, color="#ef4444", linestyle=":", alpha=0.7)
            ax_cos.axvline(float(all_arrays[cos_key].mean()), color="black", linestyle="-", alpha=0.7)
            ax_cos.set_xlim(0.5, 1.01)
        if rat_key in all_arrays:
            r = all_arrays[rat_key]
            ax_rat.hist(r, bins=60, range=(0.5, 2.5), color="#f97316", alpha=0.85)
            ax_rat.axvline(1.0, color="#ef4444", linestyle=":", alpha=0.7)
            ax_rat.axvline(float(r.mean()), color="black", linestyle="-", alpha=0.7)
            ax_rat.set_xlim(0.5, 2.5)
        ax_cos.set_ylabel(f"{oid}\n({tier})", fontsize=8)
        if row == 0:
            ax_cos.set_title("per-token cos(z_35, z_50)")
            ax_rat.set_title("per-token ||z_50|| / ||z_35||")
        if row < len(SAMPLE_ORDER) - 1:
            ax_cos.set_xticklabels([])
            ax_rat.set_xticklabels([])
    fig.suptitle("E2: Per-token direction drift — Phase III (step 35 → 50, CFG=5.0)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--tag", default="cfg5", help="Suffix for output files (e.g. cfg5, cfg7_5)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    all_arrays = {}
    for oid, _ in SAMPLE_ORDER:
        if not (args.latent_dir / oid).exists():
            print(f"[skip] {oid}: directory missing")
            continue
        rows, arrs = analyze_sample(oid, args.latent_dir)
        all_rows.extend(rows)
        all_arrays.update(arrs)
        # Short on-the-fly summary for the 35→50 pair
        for r in rows:
            if r["step_ref"] == 35 and r["step_end"] == 50:
                print(
                    f"{oid:15s}  cos_mean={r['cos_mean']:.4f}  cos_p05={r['cos_p05']:.4f}  "
                    f"ratio_mean={r['ratio_mean']:.3f}  gini={r['ratio_gini']:.3f}  "
                    f"→ {r['classification']}"
                )

    csv_path = args.out_dir / f"per_token_stats_{args.tag}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"\nWrote {csv_path}")

    hist_path = args.out_dir / f"histograms_{args.tag}.png"
    plot_histograms(all_arrays, hist_path)
    print(f"Wrote {hist_path}")

    # Aggregate summary
    pair35_50 = [r for r in all_rows if r["step_ref"] == 35 and r["step_end"] == 50]
    if pair35_50:
        cos_means = np.array([r["cos_mean"] for r in pair35_50])
        gini_vals = np.array([r["ratio_gini"] for r in pair35_50])
        print("\n=== Aggregate (step 35→50) ===")
        print(f"cos_mean across samples: min={cos_means.min():.4f} median={np.median(cos_means):.4f} max={cos_means.max():.4f}")
        print(f"gini    across samples: min={gini_vals.min():.4f} median={np.median(gini_vals):.4f} max={gini_vals.max():.4f}")
        from collections import Counter
        cls = Counter(r["classification"] for r in pair35_50)
        print(f"classifications: {dict(cls)}")


if __name__ == "__main__":
    main()
