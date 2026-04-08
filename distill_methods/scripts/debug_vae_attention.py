#!/usr/bin/env python3
"""Debug VAE attention: detailed inspection of a few samples.

Checks:
1. Processor correctness: manual softmax vs F.scaled_dot_product_attention output match
2. Per-head attention distribution (not just head-averaged)
3. Surface-near vs far-from-surface query behavior
4. Raw attention histogram at different timesteps
"""

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
OUT_DIR = PROJ / "distill_methods/reports/latent_analysis"

INSPECT_SAMPLES = ["ball_002", "chair_168", "guitar_003"]
INSPECT_STEPS = [5, 25, 30, 50]  # t=0.1, 0.5, 0.6, 1.0


class AttentionCaptureProcessor:
    def __init__(self):
        self.captured = None

    def __call__(self, attn, q, k, v):
        scale = q.shape[-1] ** -0.5
        sim = torch.matmul(q.float() * scale, k.float().transpose(-2, -1))
        weights = torch.softmax(sim, dim=-1)
        self.captured = weights.detach().cpu()
        out = torch.matmul(weights.to(v.dtype), v)
        return out


def load_pipeline():
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    return Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2.1", use_safetensors=False,
    )


def run_with_capture(vae, latent_np, query_points, device):
    """Run VAE decode and capture attention weights."""
    vae_dtype = next(vae.parameters()).dtype
    latent = torch.from_numpy(latent_np).unsqueeze(0).to(device=device, dtype=vae_dtype)
    scaled = (1.0 / vae.scale_factor) * latent

    with torch.no_grad():
        decoded = vae(scaled)

    capture = AttentionCaptureProcessor()
    vae.geo_decoder.set_cross_attention_processor(capture)

    query_t = torch.from_numpy(query_points).unsqueeze(0).to(device=device, dtype=vae_dtype)

    with torch.no_grad():
        occ = vae.geo_decoder(queries=query_t, latents=decoded)

    from hy3dshape.models.autoencoders.attention_processors import CrossAttentionProcessor
    vae.geo_decoder.cross_attn_decoder.attn.attention.attn_processor = CrossAttentionProcessor()

    return capture.captured, occ  # weights: (1, heads, N_q, N_kv), occ: (1, N_q, 1)


def run_without_capture(vae, latent_np, query_points, device):
    """Run with default processor (F.scaled_dot_product_attention) for comparison."""
    vae_dtype = next(vae.parameters()).dtype
    latent = torch.from_numpy(latent_np).unsqueeze(0).to(device=device, dtype=vae_dtype)
    scaled = (1.0 / vae.scale_factor) * latent

    with torch.no_grad():
        decoded = vae(scaled)
        occ = vae.geo_decoder(queries=torch.from_numpy(query_points).unsqueeze(0).to(device=device, dtype=vae_dtype),
                              latents=decoded)
    return occ


def main():
    device = torch.device("cuda")

    # Use a small set of query points for detailed inspection
    rng = np.random.RandomState(42)
    # 256 stratified points
    bounds = 1.01
    grid_res = 4
    voxel_size = 2.0 * bounds / grid_res
    points = []
    for ix in range(grid_res):
        for iy in range(grid_res):
            for iz in range(grid_res):
                low = np.array([-bounds + ix * voxel_size,
                                -bounds + iy * voxel_size,
                                -bounds + iz * voxel_size])
                pts = rng.rand(4, 3) * voxel_size + low
                points.append(pts)
    query_points = np.concatenate(points, axis=0).astype(np.float32)
    print(f"Query points: {query_points.shape}")

    print("Loading pipeline...")
    pipeline = load_pipeline()
    vae = pipeline.vae
    vae.to(device).eval()

    attn_module = vae.geo_decoder.cross_attn_decoder.attn.attention
    n_heads = attn_module.heads
    print(f"  Heads: {n_heads}")
    print(f"  Latent shape: {vae.latent_shape}")

    # ── Check 1: Processor correctness ─────────────────────────────
    print("\n=== Check 1: Processor correctness ===")
    latent_np = np.load(str(LATENT_DIR / "ball_002" / "latent_step_50.npy"))

    from hy3dshape.models.autoencoders.attention_processors import CrossAttentionProcessor

    vae_dtype = next(vae.parameters()).dtype
    latent_t = torch.from_numpy(latent_np).unsqueeze(0).to(device=device, dtype=vae_dtype)
    q_t = torch.from_numpy(query_points[:64]).unsqueeze(0).to(device=device, dtype=vae_dtype)

    # Run with capture processor
    capture = AttentionCaptureProcessor()
    vae.geo_decoder.set_cross_attention_processor(capture)
    with torch.no_grad():
        decoded = vae((1.0 / vae.scale_factor) * latent_t)
        occ_manual = vae.geo_decoder(queries=q_t, latents=decoded)
    # Restore with instance (not class)
    vae.geo_decoder.cross_attn_decoder.attn.attention.attn_processor = CrossAttentionProcessor()

    # Run with default processor
    with torch.no_grad():
        decoded2 = vae((1.0 / vae.scale_factor) * latent_t)
        occ_default = vae.geo_decoder(queries=q_t, latents=decoded2)

    diff = (occ_manual - occ_default).abs()
    print(f"  Output diff (manual vs default): max={diff.max().item():.6f}, mean={diff.mean().item():.6f}")
    if diff.max().item() < 0.01:
        print("  ✓ Processor outputs match")
    else:
        print("  ✗ WARNING: significant difference!")

    # ── Check 2-4: Detailed per-sample analysis ────────────────────
    for oid in INSPECT_SAMPLES:
        print(f"\n{'='*60}")
        print(f"Sample: {oid}")

        fig_hist, axes_hist = plt.subplots(len(INSPECT_STEPS), 1, figsize=(12, 3*len(INSPECT_STEPS)))
        fig_head, axes_head = plt.subplots(len(INSPECT_STEPS), 1, figsize=(12, 3*len(INSPECT_STEPS)))

        for idx, step in enumerate(INSPECT_STEPS):
            t = step / 50.0
            latent_path = LATENT_DIR / oid / f"latent_step_{step:02d}.npy"
            if not latent_path.exists():
                print(f"  t={t:.1f}: MISSING")
                continue

            latent_np = np.load(str(latent_path))
            weights, occ = run_with_capture(vae, latent_np, query_points, device)
            # weights: (1, heads, N_query, N_kv)
            w = weights[0]  # (heads, N_query, N_kv)
            occ_vals = occ[0, :, 0].cpu().numpy()  # (N_query,) occupancy predictions

            # Classify queries by occupancy
            surface_mask = (occ_vals > -0.5) & (occ_vals < 0.5)
            inside_mask = occ_vals >= 0.5
            outside_mask = occ_vals <= -0.5
            n_surface = surface_mask.sum()
            n_inside = inside_mask.sum()
            n_outside = outside_mask.sum()

            print(f"\n  t={t:.1f} (step {step}):")
            print(f"    Queries: {n_surface} surface, {n_inside} inside, {n_outside} outside")

            # Per-head entropy
            head_entropies = []
            for h in range(w.shape[0]):
                wh = w[h]  # (N_query, N_kv)
                ent = -(wh * torch.log(wh + 1e-10)).sum(dim=-1)  # (N_query,)
                head_entropies.append(ent.numpy())
            head_entropies = np.array(head_entropies)  # (heads, N_query)

            print(f"    Per-head mean entropy: {['%.2f' % h for h in head_entropies.mean(axis=1)]}")
            print(f"    Per-head eff_tokens:   {['%.0f' % np.exp(h) for h in head_entropies.mean(axis=1)]}")

            # Surface vs outside entropy (head-averaged)
            w_mean = w.mean(dim=0)  # (N_query, N_kv)
            ent_all = -(w_mean * torch.log(w_mean + 1e-10)).sum(dim=-1).numpy()
            if n_surface > 0:
                print(f"    Entropy (surface queries):  mean={ent_all[surface_mask].mean():.2f}, "
                      f"eff_tokens={np.exp(ent_all[surface_mask].mean()):.0f}")
            if n_outside > 0:
                print(f"    Entropy (outside queries):  mean={ent_all[outside_mask].mean():.2f}, "
                      f"eff_tokens={np.exp(ent_all[outside_mask].mean()):.0f}")

            # Top-k mass breakdown
            for k in [10, 50, 100]:
                topk_vals, _ = torch.topk(w_mean, k, dim=-1)
                mass = topk_vals.sum(dim=-1).numpy()
                print(f"    Top-{k} mass: all={mass.mean():.3f}", end="")
                if n_surface > 0:
                    print(f"  surface={mass[surface_mask].mean():.3f}", end="")
                if n_outside > 0:
                    print(f"  outside={mass[outside_mask].mean():.3f}", end="")
                print()

            # ── Histogram: attention weight distribution ──
            ax = axes_hist[idx]
            # Flatten all weights for histogram
            w_flat = w_mean.numpy().flatten()
            ax.hist(w_flat, bins=100, range=(0, 0.05), color="#377eb8", alpha=0.7,
                    label=f"All ({len(w_flat)} values)")
            ax.set_title(f"t={t:.1f}: Attention weight distribution (head-averaged)", fontsize=10)
            ax.set_xlabel("Attention weight")
            ax.set_ylabel("Count")
            ax.set_yscale("log")
            ax.legend(fontsize=8)

            # ── Per-head box plot ──
            ax2 = axes_head[idx]
            eff_per_head = np.exp(head_entropies.mean(axis=1))  # (heads,)
            ax2.bar(range(n_heads), eff_per_head, color="#e41a1c", alpha=0.7)
            ax2.set_title(f"t={t:.1f}: Effective tokens per head", fontsize=10)
            ax2.set_xlabel("Head")
            ax2.set_ylabel("Effective tokens (exp(H))")
            ax2.set_xticks(range(n_heads))

        fig_hist.suptitle(f"{oid}: Attention Weight Histograms", fontsize=14)
        fig_hist.tight_layout()
        fig_hist.savefig(str(OUT_DIR / f"debug_hist_{oid}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig_hist)

        fig_head.suptitle(f"{oid}: Per-Head Effective Tokens", fontsize=14)
        fig_head.tight_layout()
        fig_head.savefig(str(OUT_DIR / f"debug_perhead_{oid}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig_head)

        print(f"  Saved: debug_hist_{oid}.png, debug_perhead_{oid}.png")

    print(f"\nAll debug plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
