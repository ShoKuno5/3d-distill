# Distillation Method Comparison: Full Evaluation

**Run ID:** 20260406_1131
**Branch:** exp/distill-methods-v2 (bc7c95b)
**Date:** 2026-04-08
**Test set:** 105 samples from Toys4k (manifest_test.csv)

## Overview

Five distillation methods were trained on Hunyuan3D-2.1's flow-matching DiT, all using LoRA (rank 64) on 420 training samples for 15,000 steps. Distilled models are evaluated against the 50-step teacher on held-out test samples using geometry metrics (Chamfer distance, Hausdorff distance, F-score).

**Critical bug found and fixed during evaluation:** `resolve_model_paths` generated `checkpoints/{method}/final` but checkpoints were saved as `checkpoints/{method}/step_final`. All distilled models ran without LoRA in the initial evaluation, producing misleading results. After the fix, CD/DMD1/DMD2 went from near-total failure to 100% success.

## Methods

| Method | Mechanism | Steps | Key Idea |
|--------|-----------|-------|----------|
| PD | Progressive halving | 6 | 3-stage: 50->25->12->6, teacher at each stage is previous student |
| CD | Consistency model | 4 | Self-consistency along ODE trajectories |
| DMD1 | Regression + distribution matching | 1 | Pre-generated teacher pairs + learned score network |
| DMD2 | Adversarial + distribution matching | 1 | GAN discriminator (no regression pairs) |
| SiD | Score implicit distillation | 1 | Teacher score guidance + adaptive gradient norm |

All distilled models use `guidance_scale=1.0` (classifier-free guidance disabled) and `octree_resolution=384`.

## Training

| Method | Steps | Duration | Final Loss | Loss Components |
|--------|-------|----------|------------|-----------------|
| PD | 3 x 5,000 | 4h 37m | 0.0095 | MSE only |
| CD | 15,000 | 5h 1m | 0.0058 | consistency loss |
| DMD1 | 15,000 | 10h 43m | 0.253 | distill=0.160, regress=0.094, fake_score=1.509 |
| DMD2 | 15,000 | 36h 40m | 0.476 | distill=0.025, gan=4.503, d=0.010, fake_score=1.614 |
| SiD | 15,000 | 8h 26m | 0.118 | sid=0.118, fake_score=1.413 |

Notes:
- DMD2 training took 7x longer than CD due to 5:1 discriminator update ratio and replay buffer overhead.
- DMD2's `loss_gan=4.503` is high, indicating the generator never fully satisfied the discriminator.
- SiD loss showed a spike at step ~14,000 (0.1 -> 1.0) before recovering to 0.12, suggesting training instability.

## Inference Results

### Success Rates

| Model | Steps | Success | Failed | Rate |
|-------|-------|---------|--------|------|
| teacher_50step | 50 | 105 | 0 | 100% |
| cd_4step | 4 | 105 | 0 | 100% |
| dmd1_1step | 1 | 105 | 0 | 100% |
| dmd2_1step | 1 | 105 | 0 | 100% |
| pd_6step | 6 | 101 | 4 | 96% |
| sid_1step | 1 | 0 | 105 | 0% |

PD's 4 failures: giraffe_014, monitor_034, radio_017, tv_016 (marching cubes: empty volume).
SiD: LoRA loaded correctly but all outputs fail surface extraction. The model produces latents where the volume decoder cannot find a valid isosurface.

### Runtime

| Model | Steps | Avg Time | Diffusion | Volume Decode |
|-------|-------|----------|-----------|---------------|
| teacher_50step | 50 | 15.8s | ~4.6s | ~9s |
| pd_6step | 6 | 11.1s | ~0.5s | ~9s |
| cd_4step | 4 | 11.8s | ~0.4s | ~9s |
| dmd1_1step | 1 | 11.2s | ~0.1s | ~9s |
| dmd2_1step | 1 | 12.0s | ~0.1s | ~9s |

Volume decoding (~9s) dominates wall-clock time. The diffusion step reduction (50x for DMD1) saves only ~4.5s. True speedup requires accelerating the VAE decoder (e.g., FlashVDM approach).

## Fidelity Evaluation (vs Teacher)

### Summary (N=105 test samples)

| Model | Steps | CD (x1e-3) | Hausdorff | F@1% | F@2% |
|-------|-------|------------|-----------|------|------|
| **cd_4step** | 4 | **19.3** | **0.237** | **0.220** | **0.494** |
| dmd1_1step | 1 | 23.5 | 0.254 | 0.200 | 0.453 |
| pd_6step | 6 | 43.3 | 0.368 | 0.142 | 0.297 |
| dmd2_1step | 1 | 260.9 | 0.699 | 0.019 | 0.039 |

CD is best overall. DMD1 is close behind with 1 step (vs CD's 4). PD is third despite using the most steps. DMD2 is substantially worse.

### Distribution of Chamfer Distance

Median values reveal that the mean is skewed by a few hard samples:

| Model | CD Mean | CD Median | Ratio |
|-------|---------|-----------|-------|
| cd_4step | 0.0193 | 0.0020 | 9.7x |
| dmd1_1step | 0.0235 | 0.0032 | 7.3x |
| pd_6step | 0.0433 | 0.0191 | 2.3x |
| dmd2_1step | 0.2609 | 0.2472 | 1.1x |

CD and DMD1 have very low medians (0.002-0.003), meaning most samples are excellent. The mean is inflated by a long tail of hard cases. DMD2's mean and median are both high -- it fails uniformly.

### Hardest Samples (cd_4step)

| Object | CD | Hausdorff | F@2% |
|--------|-----|-----------|------|
| bicycle_011 | 0.276 | 0.859 | 0.043 |
| horse_027 | 0.213 | 0.714 | 0.042 |
| piano_035 | 0.202 | 0.706 | 0.091 |
| pig_019 | 0.198 | 0.753 | 0.067 |
| deer_moose_059 | 0.180 | 0.732 | 0.086 |

These are objects with thin structures (bicycle spokes, horse legs) or complex topology (piano keys). The distilled model struggles more with these than the teacher.

### DMD1 vs CD: Per-Sample Comparison

Categories where DMD1 (1-step) outperforms CD (4-step):

| Category | CD (cd_4step) | CD (dmd1_1step) | Winner |
|----------|--------------|-----------------|--------|
| hammer | 0.01054 | 0.00021 | DMD1 (50x better) |
| saw | 0.00078 | 0.00021 | DMD1 (3.7x) |
| horse | 0.21256 | 0.00095 | DMD1 (224x) |
| piano | 0.20240 | 0.00116 | DMD1 (175x) |
| deer_moose | 0.17999 | 0.00132 | DMD1 (137x) |
| candy | 0.02789 | 0.00065 | DMD1 (43x) |

Categories where CD outperforms DMD1:

| Category | CD (cd_4step) | CD (dmd1_1step) | Winner |
|----------|--------------|-----------------|--------|
| screwdriver | 0.02201 | 0.44147 | CD (20x better) |
| cat | 0.00525 | 0.24468 | CD (47x) |
| dog | 0.00065 | 0.19516 | CD (301x) |
| helicopter | 0.04441 | 0.22273 | CD (5x) |
| whale | 0.00117 | 0.09988 | CD (85x) |
| car | 0.00107 | 0.10133 | CD (95x) |

Both methods have outliers where they perform much worse than the other. This suggests complementary failure modes rather than one method being strictly dominant.

## Key Findings

1. **CD (4-step) achieves highest fidelity.** Lowest CD, Hausdorff, and highest F-score across 105 samples. 100% mesh generation success.

2. **DMD1 (1-step) is the best single-step method.** Only 22% worse than CD on mean CD, with comparable median. The regression pairs provide a strong learning signal that the adversarial-only DMD2 lacks.

3. **PD (6-step) underperforms despite more steps.** The progressive halving approach (50->25->12->6) accumulates error across stages. Total training time (3 x 5000 steps) is comparable but the cascaded architecture limits quality.

4. **DMD2 (1-step) produces valid meshes but poor quality.** The adversarial loss alone (without regression anchoring) is insufficient. `loss_gan=4.5` at end of training confirms the generator never satisfies the discriminator. Training took 36h (7x longer than CD) for the worst result.

5. **SiD completely fails.** Despite LoRA loading correctly and loss converging to 0.12, all 105 samples produce volumes where marching cubes cannot extract a surface. The score-implicit approach does not transfer to this latent space without modification.

6. **Volume decoding is the bottleneck.** All models spend ~9s on VAE decoding vs 0.1-4.6s on diffusion. Step reduction alone gives modest wall-clock improvement (4.5s / 15.8s = 28%).

## LoRA Path Bug Impact

The initial evaluation (before fix) showed catastrophic results because no LoRA weights were loaded:

| Model | Before Fix (teacher weights) | After Fix (distilled weights) |
|-------|-----|------|
| cd_4step | 9/105 success | **105/105 success** |
| dmd1_1step | 3/105 success | **105/105 success** |
| dmd2_1step | 3/105 success | **105/105 success** |

The "before" numbers reflect the teacher model running at reduced step counts (4, 1, 1) without proper denoising -- almost all outputs were empty volumes.

**Root cause:** `save_checkpoint("final")` creates `step_final/` (via `f"step_{step}"` formatting), but `resolve_model_paths` looked for `final/`. Fixed by changing the lookup to `step_final`.

## Next Steps

- **Per-category deep dive:** Analyze which shape categories each method handles best/worst. The CD vs DMD1 comparison shows complementary strengths.
- **Teacher step-quality curve:** Run teacher at 1, 2, 4, 6, 12, 25, 50 steps to establish baseline degradation without distillation. This quantifies how much distillation adds beyond simple step reduction.
- **ODE trajectory analysis:** Measure trajectory straightness in latent space. If trajectories are nearly linear, it explains why 1-step methods work well.
- **SiD debugging:** Investigate why the model produces out-of-range volume values. Check latent space statistics at inference vs training.
- **VAE acceleration:** The volume decoder bottleneck limits practical speedup. Consider FlashVDM-style decoder distillation or lower octree resolution.
