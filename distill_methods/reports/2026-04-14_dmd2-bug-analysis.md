# DMD2 Bug Analysis and Fix Experiments

**Series:** Distillation Method Comparison — Part 2
**Previous:** [2026-04-08 Distillation Method Comparison](2026-04-08_distillation-method-comparison.md)
**Date:** 2026-04-14
**Branch:** exp/cond-init (8134334)

## Motivation

Part 1 showed DMD2 achieving Chamfer Distance 260.9e-3 — 13.5x worse than CD (19.3e-3) and 11x worse than DMD1 (23.5e-3). This report investigates why DMD2 failed and implements fixes.

## Investigation Summary

### Diag-1/2: Discriminator Architecture Test

**Hypothesis:** Mean pooling over 4096 VecSet tokens destroys structural information, making the discriminator blind.

**Method:** Extracted mu_fake features for 50 real (GT latent) and 50 fake (student 1-step) samples. Trained fresh classifiers on mean-pooled vs token-level features.

**Result:**

| Classifier | Test Accuracy |
|-----------|--------------|
| Mean pool (current D design) | **100.0%** |
| Token sample (64 tokens) | 96.7% |
| Token stats (mean+std+max+min) | 100.0% |
| Random noise baseline | 100.0% |

**Conclusion:** Mean pooling retains sufficient discriminative information. **Architecture is NOT the bottleneck.** The discriminator CAN distinguish real from fake — the problem is in training dynamics.

### Training Log Analysis

DMD2 loss progression (from run 20260406_1131):

| Phase | Steps | loss_distill | loss_gan | loss_d |
|-------|-------|-------------|----------|--------|
| Early | 50-500 | 0.4-0.8 | 7-34 | 0.000-0.600 |
| Mid | 500-5000 | 0.1-0.5 | 12-30 | 0.000-0.600 |
| Late | 5000-15000 | 0.02-0.08 | 3-8 | 0.000-0.400 |

Key observations:
1. `loss_distill` converges normally (0.4 → 0.03) — KL component works
2. `loss_gan` oscillates wildly (3-34), never converges
3. `loss_d` alternates between ~0 (D overfits) and ~0.5 (D collapses)
4. Effective loss ratio: `0.03 + 0.1×5.0 = 0.53` — **GAN dominates 94% of total loss**

Compare with DMD1 (successful): loss stable at 0.1-0.3, no oscillation.

## Bugs Found (verified against arXiv:2405.14867)

### Bug 1: Replay buffer conditioning mismatch

**Paper:** Stores (image, caption) pairs. Uses correct conditioning when extracting features for real samples.

**Implementation:** `ReplayBuffer.add(x_data)` stores latent only, without `image_cond`. In `_train_discriminator`, real samples from the buffer are processed with the **current batch's** image conditioning — which belongs to a different object.

**Impact:** Discriminator learns to detect conditioning match/mismatch (shortcut learning) instead of shape quality. D easily achieves low loss by exploiting the mismatch signal, but provides no useful gradient for improving generation quality.

### Bug 2: mu_fake does not receive GAN classification gradients

**Paper** (Section 4.6): "optimizing the fake score estimator using both a denoising score matching objective on the fake data, **and the GAN classification loss**"

In the paper's UNet implementation, the discriminator is a "classification branch on top of the bottleneck of the fake diffusion denoiser" — architecturally integrated. Training D automatically updates mu_fake's upstream encoder features.

**Implementation:** Discriminator is a separate MLP. Features are extracted inside `torch.no_grad()` and `.detach()`ed. mu_fake never receives GAN gradients — only DSM gradients from `_train_fake_score`.

**Impact:** mu_fake's feature space is not shaped by the GAN objective. The discriminator operates on suboptimal features that were never trained for discrimination.

### Hyperparameter Issues

| Parameter | Original | Fixed | Rationale |
|-----------|----------|-------|-----------|
| `d_update_ratio` | 5 | 1 | Same batch reused 5x → D memorization |
| `lambda_gan` | 0.1 | 0.01 | GAN loss magnitude (~5.0) dominates KL (~0.03) at 94:6 ratio |
| `lr_d` | 5e-5 | 1e-5 | 5x student lr causes D to overfit |

### Verified Correct (no bugs)

- Flow matching convention: `x_t = t*data + (1-t)*noise` ✓
- Denoiser output: `mu = x_t + (1-t)*v` ✓
- KL loss with stop-gradient trick and adaptive weighting ✓
- GAN loss formulation (both D and G sides) ✓
- Fake score DSM training ✓
- Teacher CFG=7.5, student/fake_score no CFG ✓
- Replay buffer stores real data (not teacher outputs) — matches paper ✓

## Fix Implementation

### Bug 1 Fix: ReplayBuffer stores (latent, image_cond) pairs

```python
class ReplayBuffer:
    def add(self, x, cond):
        for i in range(x.shape[0]):
            self.buffer.append((x[i].detach().cpu(), cond[i].detach().cpu()))

    def sample(self, batch_size, device):
        indices = torch.randint(0, len(self.buffer), (batch_size,))
        xs = torch.stack([self.buffer[i][0] for i in indices]).to(device)
        conds = torch.stack([self.buffer[i][1] for i in indices]).to(device)
        return xs, conds
```

D training now uses `contexts_real = {"main": cond_real}` for real samples.

### Bug 2 Fix: mu_fake receives D gradients

Removed `torch.no_grad()` wrapper and `.detach()` from feature extraction in `_train_discriminator`. Added `optimizer_fake.step()` after D loss backward to update mu_fake with GAN classification gradients.

## Experiments (running)

| Exp | GPU | Config | Bug Fixes | Hyperparam Changes |
|-----|-----|--------|-----------|-------------------|
| Exp-1 | 0 | Bug 1+2 fixed | ✓ | d_update=1, λ_gan=0.01, lr_d=1e-5 |
| Exp-2 | 1 | Bug 1+2 fixed + regression | ✓ | Same + λ_reg=1.0, batch_size=2 |
| Exp-3 | 2 | Bug 1+2 fixed, no GAN | ✓ | λ_gan=0.0 (ablation) |

**Started:** 2026-04-14 14:04 CST
**Expected completion:** ~7h (Exp-1,3), ~14h (Exp-2 with batch_size=2)
**Success criterion:** CD < 50e-3 (from 260.9e-3)

## Results

*(To be filled after experiments complete)*

| Exp | CD (mean) | CD (median) | F@2% | Success Rate | loss_d stability |
|-----|-----------|-------------|------|-------------|-----------------|
| Exp-1 | — | — | — | — | — |
| Exp-2 | — | — | — | — | — |
| Exp-3 | — | — | — | — | — |
| Original DMD2 | 260.9e-3 | 247.2e-3 | 0.039 | 100% | 6-order oscillation |
| DMD1 (reference) | 23.5e-3 | 3.2e-3 | 0.453 | 100% | N/A |
| CD (reference) | 19.3e-3 | 2.0e-3 | 0.494 | 100% | N/A |

## Other Changes Made During Investigation

1. **Checkpoint saving fixed:** `save_checkpoint` now saves discriminator weights (`discriminator.pt`), `optimizer_d`, and `optimizer_fake` for DMD2/DMD1/SiD.
2. **Repository cleanup:** Old experiment results archived from `results/` to `archive/results/`. Stale checkpoints from failed runs archived to `archive/results/distill_methods_stale/`. `experiments/` moved to `archive/experiments/`.
3. **Diagnostic script:** `scripts/diagnose_discriminator.py` — reusable tool for testing discriminator architecture capacity.
