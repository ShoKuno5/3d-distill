# Distill Comparison: MDT-dist vs FlashVDM

## Purpose

Reproduce the comparison from the MDT-dist paper (arXiv:2509.04406): TRELLIS v1
distillation (MDT-dist) vs Hunyuan3D-2.0 distillation (FlashVDM) on Toys4k.

Four configurations are evaluated:

| Name | Base Model | Steps | Notes |
|------|-----------|-------|-------|
| trellis | TRELLIS v1 (teacher) | 25x2 | Original 25-step sampler |
| mdt_dist | TRELLIS v1 + distilled weights | 2x2 | MDT-dist 2-step distillation |
| hunyuan3d | Hunyuan3D-2.0 (teacher) | 30 | Original DiT pipeline |
| flashvdm | Hunyuan3D-2.0 + turbo DiT + FlashVDM decoder | 5 | FlashVDM 5-step distillation |

## Metrics

- **Geometry**: Chamfer Distance, F-score, Hausdorff Distance (existing pipeline)
- **Distributional**: FD_inception, FD_dinov2 (Frechet Distance on multiview renders)
- ULIP_I deferred to future experiment

## Scale

- Phase 1: 200 samples (stratified across categories)
- Phase 2: Full ~4000 samples (if results are stable)

## Paper Reference

- MDT-dist: "Multi-step Distillation of Diffusion Models via Moment Matching"
  (arXiv:2509.04406)
- FlashVDM: Built into Hunyuan3D-2.0 turbo pipeline

## Deviations from Paper

- We use Toys4k renders (512px, Cycles Blender) rather than the paper's input images
- Geometry metrics use our ICP alignment pipeline (may differ from paper's protocol)
- FD is computed on 4-view renders of predicted meshes (not on the 3D representations)

## Running

```bash
bash experiments/distill_comparison/run.sh                # Full pipeline
bash experiments/distill_comparison/run.sh --skip-inference  # Eval only
bash experiments/distill_comparison/run.sh --max-samples 5   # Smoke test
```
