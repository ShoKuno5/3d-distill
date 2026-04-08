#!/usr/bin/env python3
"""Analyze VAE cross-attention patterns across ODE timesteps.

Loads saved intermediate latents, runs them through the VAE decoder with
a custom attention-capturing processor, and measures how attention sparsity
and locality evolve — testing whether they change at the crystallization
point (t=0.5-0.6).

Usage:
    cd /mnt/workspace/kuno/distillation/models/hunyuan3d21/hy3dshape
    PYTHONPATH=.:../../../distill_methods/src:../../../pipeline \
      CUDA_VISIBLE_DEVICES=0 \
      ../../../envs/hunyuan3d-venv/bin/python \
      ../../../distill_methods/scripts/analyze_vae_attention.py
"""

import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ = Path("/mnt/workspace/kuno/distillation")
LATENT_DIR = PROJ / "results/distill_methods/intermediate_decode"
SUMMARY_CSV = LATENT_DIR / "summary.csv"
OUT_DIR = PROJ / "distill_methods/reports/latent_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STEPS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

SAMPLES = {
    "airplane_007": "airplane",
    "apple_007": "apple",
    "ball_002": "ball",
    "chair_168": "chair",
    "cup_056": "cup",
    "dinosaur_069": "dinosaur",
    "guitar_003": "guitar",
    "helicopter_015": "helicopter",
    "robot_016": "robot",
    "shoe_028": "shoe",
}

COLORS = {
    "airplane": "#e41a1c", "apple": "#377eb8", "ball": "#4daf4a",
    "chair": "#984ea3", "cup": "#ff7f00", "dinosaur": "#a65628",
    "guitar": "#f781bf", "helicopter": "#999999", "robot": "#66c2a5",
    "shoe": "#e6ab02",
}


# ── Attention Capture Processor ──────────────────────────────────────

class AttentionCaptureProcessor:
    """Replaces CrossAttentionProcessor to capture attention weights."""

    def __init__(self):
        self.captured = None

    def __call__(self, attn, q, k, v):
        scale = q.shape[-1] ** -0.5
        sim = torch.matmul(q.float() * scale, k.float().transpose(-2, -1))
        weights = torch.softmax(sim, dim=-1)  # (B, heads, N_query, N_kv)
        self.captured = weights.detach().cpu()
        out = torch.matmul(weights.to(v.dtype), v)
        return out


# ── Query Point Sampling ─────────────────────────────────────────────

def stratified_query_points(n_per_voxel=4, grid_res=8, bounds=1.01, seed=42):
    """Stratified sampling: grid_res^3 voxels, n_per_voxel points each."""
    rng = np.random.RandomState(seed)
    voxel_size = 2.0 * bounds / grid_res
    points = []
    for ix in range(grid_res):
        for iy in range(grid_res):
            for iz in range(grid_res):
                low = np.array([-bounds + ix * voxel_size,
                                -bounds + iy * voxel_size,
                                -bounds + iz * voxel_size])
                pts = rng.rand(n_per_voxel, 3) * voxel_size + low
                points.append(pts)
    return np.concatenate(points, axis=0).astype(np.float32)  # (N, 3)


# ── Metrics ──────────────────────────────────────────────────────────

def compute_metrics(weights, query_points):
    """Compute attention metrics from captured weights.

    Args:
        weights: (1, heads, N_query, N_kv) tensor
        query_points: (N_query, 3) numpy array

    Returns:
        dict of metric arrays
    """
    w = weights[0]  # (heads, N_query, N_kv)
    w_mean = w.mean(dim=0)  # (N_query, N_kv) — average across heads

    # Entropy: -sum(a * log(a))
    log_w = torch.log(w_mean + 1e-10)
    entropy = -(w_mean * log_w).sum(dim=-1)  # (N_query,)
    effective_tokens = torch.exp(entropy)  # (N_query,)

    # Top-k concentration
    n_kv = w_mean.shape[-1]
    topk_mass = {}
    for k in [10, 50, 100]:
        if k <= n_kv:
            topk_vals, _ = torch.topk(w_mean, k, dim=-1)
            topk_mass[k] = topk_vals.sum(dim=-1).numpy()  # (N_query,)

    # Spatial locality: Jaccard overlap of top-100 tokens for query pairs
    # Sample 200 query pairs at various distances
    _, top100_idx = torch.topk(w_mean, min(100, n_kv), dim=-1)  # (N_query, 100)
    top100_sets = [set(top100_idx[i].numpy()) for i in range(len(top100_idx))]

    n_q = len(query_points)
    rng = np.random.RandomState(0)
    n_pairs = min(500, n_q * (n_q - 1) // 2)
    pair_i = rng.randint(0, n_q, n_pairs)
    pair_j = rng.randint(0, n_q, n_pairs)
    # Avoid same-point pairs
    mask = pair_i != pair_j
    pair_i, pair_j = pair_i[mask], pair_j[mask]

    spatial_dists = np.linalg.norm(query_points[pair_i] - query_points[pair_j], axis=1)
    jaccards = np.array([
        len(top100_sets[i] & top100_sets[j]) / len(top100_sets[i] | top100_sets[j])
        for i, j in zip(pair_i, pair_j)
    ])

    return {
        "entropy": entropy.numpy(),
        "effective_tokens": effective_tokens.numpy(),
        "topk_mass": topk_mass,
        "locality_dists": spatial_dists,
        "locality_jaccards": jaccards,
    }


# ── Main Analysis ────────────────────────────────────────────────────

def load_pipeline():
    """Load the Hunyuan3D-2.1 pipeline (VAE only needed)."""
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2.1", use_safetensors=False,
    )
    return pipeline


def analyze_latent(pipeline, latent_np, query_points_np, device):
    """Run a single latent through VAE decoder and capture attention."""
    vae = pipeline.vae
    vae_dtype = next(vae.parameters()).dtype

    latent = torch.from_numpy(latent_np).unsqueeze(0).to(device=device, dtype=vae_dtype)
    scaled = (1.0 / vae.scale_factor) * latent
    decoded = vae(scaled)  # post_kl + transformer → (1, 4096, width)

    # Install capture processor
    capture = AttentionCaptureProcessor()
    vae.geo_decoder.set_cross_attention_processor(capture)

    # Feed query points through geo_decoder in chunks
    query_t = torch.from_numpy(query_points_np).unsqueeze(0).to(device=device, dtype=vae_dtype)
    chunk_size = 512
    all_weights = []

    with torch.no_grad():
        for start in range(0, query_t.shape[1], chunk_size):
            chunk = query_t[:, start:start + chunk_size]
            _ = vae.geo_decoder(queries=chunk, latents=decoded)
            all_weights.append(capture.captured)

    # Restore default processor
    vae.geo_decoder.set_default_cross_attention_processor()

    weights = torch.cat(all_weights, dim=2)  # (1, heads, N_query, N_kv)
    return weights


def load_cd_data():
    """Load Chamfer distance data from summary.csv."""
    cd_data = {}  # (object_id, step) -> CD value
    with open(SUMMARY_CSV) as f:
        for row in csv.DictReader(f):
            oid = row["object_id"]
            step = int(row["step"])
            cd_val = row.get("chamfer_distance", "")
            if cd_val:
                cd_data[(oid, step)] = float(cd_val)
    return cd_data


def main():
    device = torch.device("cuda")
    query_points = stratified_query_points(n_per_voxel=4, grid_res=8)
    print(f"Query points: {query_points.shape}")

    print("Loading pipeline...")
    pipeline = load_pipeline()
    pipeline.vae.to(device).eval()

    # Print model info
    geo = pipeline.vae.geo_decoder
    attn = geo.cross_attn_decoder.attn.attention
    print(f"  VAE geo_decoder heads: {attn.heads}")
    print(f"  Latent shape: {pipeline.vae.latent_shape}")

    cd_data = load_cd_data()

    # Results storage
    results = {}  # (oid, step) -> metrics dict

    for oid in SAMPLES:
        cat = SAMPLES[oid]
        print(f"\n{'='*60}")
        print(f"Sample: {oid} ({cat})")

        for step in STEPS:
            t = step / 50.0
            latent_path = LATENT_DIR / oid / f"latent_step_{step:02d}.npy"
            if not latent_path.exists():
                print(f"  t={t:.1f}: MISSING")
                continue

            latent_np = np.load(str(latent_path))
            t0 = time.time()

            with torch.no_grad():
                weights = analyze_latent(pipeline, latent_np, query_points, device)

            metrics = compute_metrics(weights, query_points)
            elapsed = time.time() - t0

            ent = metrics["entropy"].mean()
            eff = metrics["effective_tokens"].mean()
            top10 = metrics["topk_mass"].get(10, np.array([0])).mean()

            cd_val = cd_data.get((oid, step), float("nan"))
            print(f"  t={t:.1f}: entropy={ent:.2f}  eff_tokens={eff:.0f}  "
                  f"top10={top10:.3f}  CD={cd_val:.4f}  ({elapsed:.1f}s)")

            results[(oid, step)] = metrics

    # Save raw results
    np.savez_compressed(
        str(OUT_DIR / "attention_metrics.npz"),
        **{f"{oid}_{step}_{k}": v
           for (oid, step), m in results.items()
           for k, v in m.items()
           if isinstance(v, np.ndarray)},
    )

    # ── Plotting ─────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_entropy_vs_timestep(results, cd_data)
    plot_effective_tokens(results)
    plot_topk_concentration(results)
    plot_locality_grid(results)
    plot_combined_crystallization(results, cd_data)
    print(f"\nAll plots saved to {OUT_DIR}/")


# ── Plotting Functions ───────────────────────────────────────────────

def plot_entropy_vs_timestep(results, cd_data):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    t_vals = [s / 50.0 for s in STEPS]

    for oid in SAMPLES:
        cat = SAMPLES[oid]
        ents = []
        for step in STEPS:
            m = results.get((oid, step))
            ents.append(m["entropy"].mean() if m else np.nan)
        ax1.plot(t_vals, ents, "-o", color=COLORS[cat], label=cat,
                 markersize=4, alpha=0.7)

    ax1.axvspan(0.5, 0.7, alpha=0.1, color="red", label="Crystallization zone")
    ax1.axhline(np.log(4096), color="gray", linestyle="--", alpha=0.5, label=f"Max entropy (log 4096 = {np.log(4096):.2f})")
    ax1.set_xlabel("t (ODE progress)")
    ax1.set_ylabel("Mean Attention Entropy")
    ax1.set_title("VAE Cross-Attention Entropy Across ODE Steps")
    ax1.legend(fontsize=7, ncol=3)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(OUT_DIR / "attention_entropy_vs_timestep.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: attention_entropy_vs_timestep.png")


def plot_effective_tokens(results):
    fig, ax = plt.subplots(figsize=(10, 6))
    t_vals = [s / 50.0 for s in STEPS]

    all_curves = []
    for oid in SAMPLES:
        cat = SAMPLES[oid]
        effs = []
        for step in STEPS:
            m = results.get((oid, step))
            effs.append(m["effective_tokens"].mean() if m else np.nan)
        ax.plot(t_vals, effs, "-o", color=COLORS[cat], label=cat,
                markersize=4, alpha=0.7)
        all_curves.append(effs)

    # Mean curve
    mean_curve = np.nanmean(all_curves, axis=0)
    ax.plot(t_vals, mean_curve, "k-s", linewidth=2.5, markersize=6, label="Mean", zorder=10)

    ax.axvspan(0.5, 0.7, alpha=0.1, color="red", label="Crystallization zone")
    ax.axhline(10, color="blue", linestyle="--", alpha=0.5, label="FlashVDM: ~10 tokens")
    ax.set_xlabel("t (ODE progress)")
    ax.set_ylabel("Effective Number of Attended Tokens (exp(H))")
    ax.set_title("Attention Sparsity: How Many Tokens Matter?")
    ax.set_yscale("log")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(OUT_DIR / "effective_tokens_vs_timestep.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: effective_tokens_vs_timestep.png")


def plot_topk_concentration(results):
    fig, ax = plt.subplots(figsize=(10, 6))
    t_vals = [s / 50.0 for s in STEPS]

    for k_val, style in [(10, "-o"), (50, "-s"), (100, "-^")]:
        means = []
        for step in STEPS:
            vals = []
            for oid in SAMPLES:
                m = results.get((oid, step))
                if m and k_val in m["topk_mass"]:
                    vals.append(m["topk_mass"][k_val].mean())
            means.append(np.mean(vals) if vals else np.nan)
        ax.plot(t_vals, means, style, label=f"Top-{k_val}", linewidth=2, markersize=5)

    ax.axvspan(0.5, 0.7, alpha=0.1, color="red", label="Crystallization zone")
    ax.set_xlabel("t (ODE progress)")
    ax.set_ylabel("Fraction of Attention Mass in Top-k")
    ax.set_title("Attention Concentration Over ODE Steps")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(OUT_DIR / "topk_concentration_vs_timestep.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: topk_concentration_vs_timestep.png")


def plot_locality_grid(results):
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    fig.suptitle("Spatial Locality of Attention (Jaccard overlap vs 3D distance)", fontsize=16, y=1.02)

    for idx, step in enumerate(STEPS):
        ax = axes[idx // 5, idx % 5]
        t = step / 50.0

        all_dists = []
        all_jacs = []
        for oid in SAMPLES:
            m = results.get((oid, step))
            if m:
                all_dists.append(m["locality_dists"])
                all_jacs.append(m["locality_jaccards"])

        if not all_dists:
            ax.set_title(f"t={t:.1f} (no data)")
            continue

        dists = np.concatenate(all_dists)
        jacs = np.concatenate(all_jacs)

        # Bin by distance
        n_bins = 10
        bins = np.linspace(0, dists.max() + 1e-8, n_bins + 1)
        bin_centers = []
        bin_means = []
        bin_stds = []
        for i in range(n_bins):
            mask = (dists >= bins[i]) & (dists < bins[i + 1])
            if mask.sum() > 5:
                bin_centers.append((bins[i] + bins[i + 1]) / 2)
                bin_means.append(jacs[mask].mean())
                bin_stds.append(jacs[mask].std())

        if bin_centers:
            ax.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt="-o",
                        color="#e41a1c", markersize=4, capsize=3, linewidth=1.5)

        ax.set_title(f"t={t:.1f} (step {step})", fontsize=11)
        ax.set_xlabel("3D distance")
        ax.set_ylabel("Jaccard overlap (top-100)")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(OUT_DIR / "attention_locality_grid.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: attention_locality_grid.png")


def plot_combined_crystallization(results, cd_data):
    """The key figure: overlay CD curve with attention metrics."""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    t_vals = [s / 50.0 for s in STEPS]

    # CD curve (mean across samples, log scale)
    cd_means = []
    for step in STEPS:
        vals = [cd_data.get((oid, step), np.nan) for oid in SAMPLES]
        vals = [v for v in vals if not np.isnan(v)]
        cd_means.append(np.mean(vals) if vals else np.nan)

    color_cd = "#e41a1c"
    ax1.plot(t_vals, cd_means, "s-", color=color_cd, linewidth=2.5, markersize=7, label="Chamfer Distance")
    ax1.set_yscale("log")
    ax1.set_ylabel("Chamfer Distance (log)", color=color_cd, fontsize=12)
    ax1.tick_params(axis="y", labelcolor=color_cd)
    ax1.set_xlabel("t (ODE progress)", fontsize=12)

    # Effective tokens (right axis)
    ax2 = ax1.twinx()
    eff_means = []
    for step in STEPS:
        vals = []
        for oid in SAMPLES:
            m = results.get((oid, step))
            if m:
                vals.append(m["effective_tokens"].mean())
        eff_means.append(np.mean(vals) if vals else np.nan)

    color_eff = "#377eb8"
    ax2.plot(t_vals, eff_means, "o-", color=color_eff, linewidth=2.5, markersize=7, label="Effective Tokens")
    ax2.set_yscale("log")
    ax2.set_ylabel("Effective Attended Tokens (log)", color=color_eff, fontsize=12)
    ax2.tick_params(axis="y", labelcolor=color_eff)

    # Crystallization zone
    ax1.axvspan(0.5, 0.7, alpha=0.1, color="red")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=10)

    ax1.set_title("Structure Crystallization: Geometry Quality vs Attention Sparsity", fontsize=14)
    ax1.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(str(OUT_DIR / "combined_crystallization.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: combined_crystallization.png")


if __name__ == "__main__":
    main()
