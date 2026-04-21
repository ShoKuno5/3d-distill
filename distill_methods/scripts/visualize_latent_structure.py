#!/usr/bin/env python3
"""Visualize latent space structure across ODE timesteps.

Loads intermediate ODE latents and analyzes:
1. PCA of pooled latents at each timestep (do categories cluster?)
2. Pairwise cosine similarity matrix at each timestep
3. Inter/intra category distance ratio over time (cluster separation metric)

Usage:
    python distill_methods/scripts/visualize_latent_structure.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform

LATENT_DIR = "/mnt/workspace/kuno/distillation/results/distill_methods/intermediate_decode"
OUT_DIR = "/mnt/workspace/kuno/distillation/distill_methods/reports/latent_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

STEPS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

# Sample metadata (object_id -> category)
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
    "airplane": "#e41a1c",
    "apple": "#377eb8",
    "ball": "#4daf4a",
    "chair": "#984ea3",
    "cup": "#ff7f00",
    "dinosaur": "#a65628",
    "guitar": "#f781bf",
    "helicopter": "#999999",
    "robot": "#66c2a5",
    "shoe": "#e6ab02",
}


def load_latents():
    """Load all latents. Returns dict[step][object_id] = np.array(4096, 64)."""
    latents = {}
    for step in STEPS:
        latents[step] = {}
        for oid in SAMPLES:
            path = os.path.join(LATENT_DIR, oid, f"latent_step_{step:02d}.npy")
            if os.path.exists(path):
                latents[step][oid] = np.load(path)
    return latents


def pool_latent(x):
    """Pool (4096, 64) vectset to a single vector. Try multiple strategies."""
    return {
        "mean": x.mean(axis=0),          # (64,) - average over tokens
        "std": x.std(axis=0),             # (64,) - spread of tokens
        "norm_per_token": np.linalg.norm(x, axis=1),  # (4096,) - per-token norms
        "flat_pca": None,  # filled later
    }


def plot_pca_grid(latents):
    """PCA of mean-pooled latents at each timestep, using a shared PCA basis and axis scale."""
    # Collect ALL mean-pooled vectors across all timesteps
    all_vecs = []
    all_meta = []  # (step, oid)
    for step in STEPS:
        for oid in SAMPLES:
            if oid in latents[step]:
                all_vecs.append(latents[step][oid].mean(axis=0))
                all_meta.append((step, oid))

    X_all = np.stack(all_vecs)
    pca = PCA(n_components=2)
    X_all_pca = pca.fit_transform(X_all)

    # Global axis limits
    pad = 0.05
    x_min, x_max = X_all_pca[:, 0].min(), X_all_pca[:, 0].max()
    y_min, y_max = X_all_pca[:, 1].min(), X_all_pca[:, 1].max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    xlim = (x_min - pad * x_range, x_max + pad * x_range)
    ylim = (y_min - pad * y_range, y_max + pad * y_range)

    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    fig.suptitle(
        f"PCA of Mean-Pooled VecSet Latents (shared basis, "
        f"PC1={pca.explained_variance_ratio_[0]:.0%}, PC2={pca.explained_variance_ratio_[1]:.0%})",
        fontsize=16, y=1.02,
    )

    for idx, step in enumerate(STEPS):
        ax = axes[idx // 5, idx % 5]
        t = step / 50.0

        # Extract this timestep's projected points
        for i, (s, oid) in enumerate(all_meta):
            if s == step:
                cat = SAMPLES[oid]
                ax.scatter(X_all_pca[i, 0], X_all_pca[i, 1],
                           c=COLORS[cat], s=120, zorder=5, edgecolors="black", linewidth=0.5)
                ax.annotate(cat, (X_all_pca[i, 0], X_all_pca[i, 1]),
                            fontsize=7, ha="center", va="bottom", xytext=(0, 5),
                            textcoords="offset points")

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(f"t={t:.1f} (step {step})", fontsize=11)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "pca_grid.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_cosine_similarity(latents):
    """Pairwise cosine similarity of mean-pooled latents at each timestep."""
    fig, axes = plt.subplots(2, 5, figsize=(30, 12))
    fig.suptitle("Pairwise Cosine Similarity of Mean-Pooled Latents", fontsize=16, y=1.02)

    oids = list(SAMPLES.keys())
    cats = [SAMPLES[oid] for oid in oids]

    for idx, step in enumerate(STEPS):
        ax = axes[idx // 5, idx % 5]
        t = step / 50.0

        vecs = []
        for oid in oids:
            if oid in latents[step]:
                v = latents[step][oid].mean(axis=0)
                v = v / (np.linalg.norm(v) + 1e-8)
                vecs.append(v)
            else:
                vecs.append(np.zeros(64))

        X = np.stack(vecs)
        sim = X @ X.T

        im = ax.imshow(sim, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(cats)))
        ax.set_yticks(range(len(cats)))
        ax.set_xticklabels(cats, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(cats, fontsize=7)
        ax.set_title(f"t={t:.1f} (step {step})", fontsize=11)

        # Annotate values
        for i in range(len(cats)):
            for j in range(len(cats)):
                ax.text(j, i, f"{sim[i,j]:.2f}", ha="center", va="center", fontsize=5,
                        color="white" if abs(sim[i,j]) > 0.5 else "black")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "cosine_similarity_grid.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_distance_evolution(latents):
    """Track pairwise distances over timesteps to measure cluster formation."""
    oids = list(SAMPLES.keys())

    # Mean-pooled L2 distances at each step
    mean_dists = []
    std_dists = []
    min_dists = []
    max_dists = []

    # Also track: spread within each sample's tokens (intra-sample variance)
    token_spreads = []

    for step in STEPS:
        vecs = []
        spreads = []
        for oid in oids:
            if oid in latents[step]:
                x = latents[step][oid]  # (4096, 64)
                vecs.append(x.mean(axis=0))
                spreads.append(x.std())

        X = np.stack(vecs)
        dists = pdist(X, metric="euclidean")
        mean_dists.append(dists.mean())
        std_dists.append(dists.std())
        min_dists.append(dists.min())
        max_dists.append(dists.max())
        token_spreads.append(np.mean(spreads))

    t_vals = [s / 50.0 for s in STEPS]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Pairwise distance evolution
    ax1.plot(t_vals, mean_dists, "b-o", label="Mean pairwise dist", linewidth=2)
    ax1.fill_between(t_vals,
                     [m - s for m, s in zip(mean_dists, std_dists)],
                     [m + s for m, s in zip(mean_dists, std_dists)],
                     alpha=0.2, color="blue")
    ax1.plot(t_vals, min_dists, "g--", label="Min dist", alpha=0.7)
    ax1.plot(t_vals, max_dists, "r--", label="Max dist", alpha=0.7)
    ax1.axvspan(0.5, 0.7, alpha=0.1, color="red", label="Crystallization zone")
    ax1.set_xlabel("t (ODE progress)")
    ax1.set_ylabel("L2 distance")
    ax1.set_title("Pairwise Distance Between Samples")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Max/Min ratio (separation measure)
    ratios = [mx / (mn + 1e-8) for mx, mn in zip(max_dists, min_dists)]
    ax2.plot(t_vals, ratios, "r-o", linewidth=2)
    ax2.axvspan(0.5, 0.7, alpha=0.1, color="red", label="Crystallization zone")
    ax2.set_xlabel("t (ODE progress)")
    ax2.set_ylabel("Max/Min distance ratio")
    ax2.set_title("Cluster Separation (higher = more differentiated)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Intra-sample token spread
    ax3.plot(t_vals, token_spreads, "m-o", linewidth=2)
    ax3.axvspan(0.5, 0.7, alpha=0.1, color="red", label="Crystallization zone")
    ax3.set_xlabel("t (ODE progress)")
    ax3.set_ylabel("Mean token std")
    ax3.set_title("Intra-Sample Token Spread")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "distance_evolution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_trajectory_straightness(latents):
    """Measure how straight the ODE trajectory is for each sample."""
    oids = list(SAMPLES.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for oid in oids:
        cat = SAMPLES[oid]

        # Collect mean-pooled latents along trajectory
        traj = []
        steps_valid = []
        for step in STEPS:
            if oid in latents[step]:
                traj.append(latents[step][oid].mean(axis=0))
                steps_valid.append(step)

        if len(traj) < 3:
            continue

        traj = np.stack(traj)

        # Straightness: ||x_end - x_start|| / sum(||x_{i+1} - x_i||)
        chord = np.linalg.norm(traj[-1] - traj[0])
        arc = sum(np.linalg.norm(traj[i+1] - traj[i]) for i in range(len(traj)-1))
        straightness = chord / (arc + 1e-8)

        # Cumulative arc length (normalized)
        cum_arc = [0]
        for i in range(len(traj)-1):
            cum_arc.append(cum_arc[-1] + np.linalg.norm(traj[i+1] - traj[i]))
        cum_arc = np.array(cum_arc)

        # Deviation from straight line at each point
        direction = (traj[-1] - traj[0]) / (chord + 1e-8)
        deviations = []
        for i in range(len(traj)):
            proj = np.dot(traj[i] - traj[0], direction)
            proj_point = traj[0] + proj * direction
            dev = np.linalg.norm(traj[i] - proj_point)
            deviations.append(dev)

        t_vals = [s / 50.0 for s in steps_valid]
        ax1.plot(t_vals, deviations, "-o", color=COLORS[cat], label=f"{cat} (S={straightness:.3f})",
                 markersize=4)
        ax2.bar(cat, straightness, color=COLORS[cat], edgecolor="black", linewidth=0.5)

    ax1.axvspan(0.5, 0.7, alpha=0.1, color="red")
    ax1.set_xlabel("t (ODE progress)")
    ax1.set_ylabel("Deviation from straight line")
    ax1.set_title("ODE Trajectory Deviation from Straight Path")
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.3)

    ax2.set_ylabel("Straightness (1.0 = perfectly straight)")
    ax2.set_title("Trajectory Straightness per Sample")
    ax2.tick_params(axis="x", rotation=45)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "trajectory_straightness.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_tsne(latents):
    """t-SNE of all (sample, timestep) latents in a single embedding."""
    all_vecs = []
    all_labels = []
    all_steps = []

    for step in STEPS:
        for oid in SAMPLES:
            if oid in latents[step]:
                all_vecs.append(latents[step][oid].mean(axis=0))
                all_labels.append(SAMPLES[oid])
                all_steps.append(step / 50.0)

    X = np.stack(all_vecs)
    tsne = TSNE(n_components=2, perplexity=min(15, len(X)-1), random_state=42)
    X_tsne = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 8))
    for i in range(len(X_tsne)):
        cat = all_labels[i]
        t = all_steps[i]
        alpha = 0.3 + 0.7 * t  # early steps are faint, later steps are bold
        size = 30 + 100 * t
        ax.scatter(X_tsne[i, 0], X_tsne[i, 1],
                   c=COLORS[cat], s=size, alpha=alpha,
                   edgecolors="black", linewidth=0.3)

    # Draw trajectories
    for oid in SAMPLES:
        cat = SAMPLES[oid]
        indices = [i for i in range(len(all_labels))
                   if all_labels[i] == cat]
        if len(indices) > 1:
            pts = X_tsne[indices]
            ax.plot(pts[:, 0], pts[:, 1], "-", color=COLORS[cat], alpha=0.4, linewidth=1)

    # Legend
    for cat, color in COLORS.items():
        ax.scatter([], [], c=color, s=80, label=cat, edgecolors="black", linewidth=0.5)
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.set_title("t-SNE of All Latents (faint=early, bold=late)\nTrajectories show ODE path per sample")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "tsne_trajectories.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    print("Loading latents...")
    latents = load_latents()
    print(f"Loaded {sum(len(v) for v in latents.values())} latent arrays "
          f"({len(SAMPLES)} samples x {len(STEPS)} steps)")

    print("\n1. PCA grid...")
    plot_pca_grid(latents)

    print("2. Cosine similarity...")
    plot_cosine_similarity(latents)

    print("3. Distance evolution...")
    plot_distance_evolution(latents)

    print("4. Trajectory straightness...")
    plot_trajectory_straightness(latents)

    print("5. t-SNE trajectories...")
    plot_tsne(latents)

    print(f"\nAll plots saved to {OUT_DIR}/")
