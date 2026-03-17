# Evaluation Protocol: 4-Way Toys4k Geometry Comparison

## Models

| ID | Model | Version | Family |
|----|-------|---------|--------|
| trellis | TRELLIS | 1.0 | Microsoft TRELLIS |
| trellis2 | TRELLIS.2 | 2.0 | Microsoft TRELLIS |
| hunyuan3d | Hunyuan3D | 2.0 | Tencent Hunyuan3D |
| hunyuan3d21 | Hunyuan3D | 2.1 | Tencent Hunyuan3D |

## Dataset

- **Source**: Toys4k
- **Task**: Single-image → 1-object mesh
- **GT**: Point clouds (10K points per object, from `pc10K.npz`)
- **Input**: Blend renders (PNG, converted to RGBA for all models)
- **Subset**: Defined per experiment (e.g. `experiments/toys4k_baseline/manifest.csv`, 10 objects, 10 categories)

## Seed Control

- **Inference seed**: 42 (all models)
- **Point sampling seed**: 0
- **Bootstrap seed**: 42
- **ICP subsampling seed**: 0
- Expandable to `seeds: [0, 1, 2]` in config

## GT Canonicalization

1. Load GT point cloud from `pc10K.npz` (Toys4k format: `data['dct'].item()['pc']`)
2. Compute bbox center: `center = (min + max) / 2`
3. Translate to origin: `pts -= center`
4. Compute max point norm: `scale = max(||pts_i||)`
5. Scale to unit sphere: `pts /= scale`

After this, all GT points satisfy `||p|| <= 1.0`.

## Prediction Mesh Cleaning

Applied identically to all models:

1. **NaN removal**: delete vertices with any NaN coordinate; delete faces referencing them
2. **Inf removal**: same for Inf
3. **Degenerate face removal**: delete faces with zero area (cross product norm < 1e-10)
4. **Unreferenced vertex removal**: delete vertices not referenced by any face
5. **Tiny component removal**: delete connected components with < 1% of total faces

No smoothing, no manual rotation, no model-specific post-processing.

## Point Cloud Sampling

- Method: `trimesh.sample.sample_surface` (area-proportional random sampling)
- Alignment cloud: 100,000 points (dense, for ICP convergence)
- Evaluation cloud: 16,384 points (for metric computation)
- Seed: 0 (fixed for reproducibility)

## Prediction Normalization

Same as GT: bbox center → origin, isotropic scale → unit sphere (max norm = 1).

## Alignment

### Track A: Similarity ICP

Two-phase multi-start approach:

**Phase 1 — Coarse search:**
- Subsample both clouds to 5,000 points
- Try all 24 proper rotations of the cube (SO(3) octahedral group)
- For each initial rotation, run similarity ICP with max 50 iterations
- Pick the rotation with lowest cost

**Phase 2 — Dense refinement:**
- From the best coarse result, run similarity ICP on the full 100K cloud
- Max 100 iterations, tolerance 1e-6

**Similarity ICP details:**
- Each iteration: find nearest neighbors, solve Umeyama alignment
- Umeyama solves for optimal (R, t, s) minimizing ||target - (s·R·source + t)||²
- Scale clamped to [0.5, 2.0] to prevent degenerate collapse
- No reflection: det(R) > 0 enforced via SVD sign correction
- Convergence: one-directional cost (source→target) for speed

**Forbidden:**
- Reflection (det(R) < 0)
- Non-uniform scale
- Manual adjustment

### Track B: Scale Only

- Both GT and prediction normalized to unit sphere (same procedure)
- No rotation alignment applied
- Identity transform used

## Metrics

### Chamfer Distance (CD)

Bilateral mean of squared L2 distances:

```
CD = (1/N) Σ ||pred_i - NN_gt(pred_i)||² + (1/M) Σ ||gt_j - NN_pred(gt_j)||²
```

where NN denotes nearest neighbor. Lower is better.

### F-score @ τ

```
precision(τ) = fraction of pred points within distance τ of any GT point
recall(τ)    = fraction of GT points within distance τ of any pred point
F(τ)         = 2 · precision · recall / (precision + recall)
```

Computed at τ = 0.01 and τ = 0.02 (in unit-sphere coordinates). Higher is better.

### Hausdorff Distance

```
HD = max(max_i d(pred_i, GT), max_j d(gt_j, Pred))
```

Maximum directed distance in both directions. Lower is better.

### Summary Statistics

- Mean, median, std over samples
- 95% bootstrap confidence interval (1000 resamples, seed=42)
- Failure count and rate
