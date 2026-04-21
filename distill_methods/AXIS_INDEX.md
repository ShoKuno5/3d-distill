# Axis Index

This project has two research axes that were historically mixed but should be
tracked separately going forward. See [STATE.md](STATE.md) for current knowledge
per axis and [FORWARD.md](FORWARD.md) for next steps.

- **Axis A** — Teacher trajectory / stepwise decoding behavior
  (*what does the VecSet teacher do?*)
- **Axis B** — Distillation comparison / student-vs-teacher
  (*how do student methods fare, and what predicts success?*)
- **Shared** — Pipeline / dataset / infrastructure used by both axes

Paths below are relative to the repo root (`/mnt/workspace/kuno/distillation/`).

---

## Axis A — Teacher trajectory & stepwise decoding

### Scripts (`distill_methods/scripts/`)

| Script | Purpose |
|---|---|
| `decode_intermediate_steps.py` | Teacher 50-step ODE with snapshots at 10 checkpoints; saves latents + decoded meshes + per-step CD |
| `analyze_phase3_drift.py` | Per-token cos-sim and norm-ratio across step pairs (E2) |
| `analyze_teacher_reference.py` | Aggregate per-token L2 norm stats from saved latents (N=105 teacher ref) |
| `_volume_logit_analysis.py` | VAE + volume_decoder helper for teacher intermediate latents |
| `analyze_vae_attention.py` | Attention entropy / locality on teacher VAE |
| `debug_vae_attention.py` | VAE attention debugging |
| `visualize_latent_structure.py` | PCA / t-SNE on teacher trajectory latents |
| `render_intermediate_steps.py` | Render intermediate step meshes |

### Configs

| File | Purpose |
|---|---|
| `distill_methods/config_teacher_extended.yaml` | Teacher 100/200-step on 3 failure-category samples (E3) |

### Results (`results/distill_methods/`)

| Path | Size | Contents |
|---|---|---|
| `intermediate_decode/` | 1.5G | 10 sample × 10 step latents + decoded meshes (CFG=7.5) |
| `intermediate_decode_cfg5/` | 1.5G | Same, CFG=5.0 replicate |
| `runs/20260421_manifold_diagnosis/e2_phase3_drift/` | light | Per-token stats CSVs + histograms |
| `runs/20260421_manifold_diagnosis/e3_teacher_extended/` | light | Teacher 100/200-step meshes + metrics |
| `runs/20260421_manifold_diagnosis/teacher_ref_n105/` | 110M | N=105 teacher step-50 latents + stats.json |

### Reports (`distill_methods/reports/`)

| Report | Notes |
|---|---|
| `reports/latent_analysis/` (PNG assets) | VAE attention plots, PCA grid, t-SNE, cosine grid |
| `2026-04-08_combined-analysis.html` (**mixed**) | Teacher ODE analysis is Axis A; method comparison is Axis B |
| `2026-04-21_manifold-diagnosis.html` (**mixed**) | E2/E3 are Axis A; E1 is Axis B |

---

## Axis B — Distillation comparison & student-vs-teacher

### Scripts (`distill_methods/scripts/`)

| Script | Purpose |
|---|---|
| `run_cross_family_benchmark.sh` | 9-model wave-based inference orchestrator |
| `generate_cross_family_report.py` | Self-contained HTML report for cross-family run |
| `generate_combined_report.py` | Older combined report generator (2026-04-08) |
| `diagnose_dmd1_manifold.py` | DMD1 with output_type=latent + volume_decoder stats (E1) |
| `diagnose_discriminator.py` | DMD2 discriminator debugging |
| `diagnose_fairness.py` | alignment_scale histogram, clamp hit rate, per-model mesh quality |
| `e1_postprocess.py` | Join E1 diagnosis with cross-family CD, mode × CD analysis, norm-vs-CD scatter |
| `compare_teacher_student.py` | Teacher-student direct comparison |
| `eval_distill.py` | Distillation-specific evaluation wrapper |
| `merge_lora_checkpoint.py` | LoRA adapter merging utility |
| `generate_dmd1_pairs.py` | Pre-compute teacher-output regression pairs for DMD1 |
| `smoke_test_pd.py`, `smoke_test_all.py` | Smoke tests for distillation methods |
| `test_inference_pd.py`, `test_stage_handoff.py`, `test_stage_handoff_lite.py` | Stage handoff tests |

### Source (`distill_methods/src/`)

All files are Axis B (distillation training code):
- `train.py`, `base_distiller.py`
- `progressive_distillation.py`, `consistency_distillation.py`
- `dmd1_distillation.py`, `dmd2_distillation.py`, `sid_distillation.py`

### Configs

| File | Purpose |
|---|---|
| `distill_methods/config.yaml` | Base distillation training config (PD/CD/DMD1/DMD2) |
| `distill_methods/config_cross_family.yaml` | 9-model inference-eval config (teacher_50step entry is Axis A reference) |

### Results (`results/distill_methods/`)

| Path | Size | Axis B sub-type |
|---|---|---|
| `runs/20260406_1131/` | 65G on CPFS | Distillation training run (4 methods × 15K steps) |
| `runs/20260414_dmd1_multistep/` | medium | DMD1 K=1,2,4 extension runs |
| `runs/20260414_dmd2_fix/` | 35G on CPFS | DMD2 bug-fix re-training |
| `runs/20260421_cross_family/` | 129G on CPFS | 9-model × 105 sample inference + evaluation |
| `runs/20260421_manifold_diagnosis/e1_dmd1_volume_logit/` | ~110M | DMD1 latents (105) + diagnosis |

> **Storage note**: heavy training runs may be migrated to `/mnt/oss/kuno/past_runs/`
> as OSS mirror. Check `/mnt/oss/kuno/` if a CPFS path is unavailable.

### Reports

| Report | Notes |
|---|---|
| `2026-04-08_distillation-method-comparison.md` | 4-method comparison table |
| `2026-04-14_dmd2-bug-analysis.md` | DMD2 bug investigation (Part 2 of the series) |
| `2026-04-21_cross-family-benchmark.html` | 9-model cross-family benchmark |
| `2026-04-21_manifold-diagnosis.html` (mixed, E1 portion) | DMD1 norm analysis |

---

## Shared infrastructure

### Pipeline (`pipeline/`)

- `scripts/run_eval.py` — Track-A (ICP) + Track-B (scale-only) evaluation with bootstrap CI
- `scripts/run_inference_distilled.py` — generic student dispatcher (used by Axis B)
- `scripts/run_inference_trellis.py`, `run_inference_trellis2.py`, `run_inference_mdt_dist.py` — per-model inference
- `scripts/run_inference_flashvdm.py` — FlashVDM-specific runner
- `src/geometry/alignment.py` — Similarity ICP + Umeyama (configurable scale_clamp)
- `src/geometry/normalize.py`, `sampling.py`, `cleaning.py`
- `src/evaluation/metrics.py` — Chamfer / Hausdorff / F-score / Fréchet distance
- `src/data/toys4k.py` — manifest loader + category filter
- `src/utils/inference_config.py` — config loader with per-model defaults

### Dataset

- `distill_methods/manifest.csv`, `manifest_train.csv`, `manifest_test.csv`
- `datasets/Toys4k/` (gitignored; official/ + renders/ + zips/ + compat/)

### Orchestration

- `distill_methods/run.sh` — distillation training driver (Axis B)
- `distill_methods/prepare_data.sh` — dataset prep (Shared)
- `distill_methods/scripts/create_manifests.py` (Shared)
- `distill_methods/scripts/prepare_training_data.py` (Shared)
- `distill_methods/scripts/render_batch.py` (Shared)
- `distill_methods/scripts/run_all_inference.sh` — wrapper orchestrator

### Misc

- `distill_methods/README.md` — project overview
- `distill_methods/KNOWN_BUGS.md` — historical bug log
- `distill_methods/src/__init__.py`

---

## Slightly-mixed scripts (noted explicitly)

- `e3_postprocess.py` — E3 compares teacher CDs at 50/100/200 step. The
  underlying *measurement* is Axis A (teacher behavior), but the script's
  analysis format mirrors cross-family Axis B scripts. Filed as Axis A.
- `compare_teacher_student.py` — generic comparison, used by both but
  primarily for Axis B.
- `generate_combined_report.py` — produces the mixed combined-analysis
  report. Axis B by code structure but outputs mixed content.
