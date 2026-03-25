# Distill Profiling

Unified profiling + quality evaluation for distilled models vs their teachers.
N=48 samples (same manifest as `inference_profiling/`).

## Models

| Model | Base | Steps | Role |
|-------|------|-------|------|
| trellis | TRELLIS v1 | 25×2 | Teacher |
| mdt_dist | TRELLIS v1 + MDT-dist | 2×2 | Distilled |
| hunyuan3d | Hunyuan3D-2.0 | 30 | Teacher |
| flashvdm | Hunyuan3D-2.0 turbo + FlashVDM | 5 | Distilled |

## What it produces

1. **Block-level timing** — `profile.json` per sample per model, summary CSVs
2. **Mesh predictions** — `mesh_raw.obj` per sample per model
3. **Geometry metrics** — Chamfer Distance, F-score, Hausdorff (ICP-aligned)
4. **Frechet Distance** — InceptionV3 + DINOv2 on multiview renders
5. **Report** — Markdown summary

## Usage

```bash
bash experiments/distill_profiling/run.sh            # Full run (4 GPUs)
bash experiments/distill_profiling/run.sh --skip-inference --workers=4  # Eval only
```
