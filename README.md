# 3D Generation Model Evaluation

Quantitative benchmark comparing single-image 3D reconstruction models on a Toys4k subset.
Measures Chamfer Distance, F-score, Hausdorff Distance, and other geometry metrics.

## Active Models

| Shortname | Model | Directory | Environment |
|-----------|-------|-----------|-------------|
| trellis2 | TRELLIS 2.0 | `models/trellis2/` | `envs/miniconda3/envs/trellis2/` |
| hunyuan3d21 | Hunyuan3D 2.1 | `models/hunyuan3d21/` | `envs/hunyuan3d-venv/` (shared, timm added) |

## Reference-Only Models

| Shortname | Model | Directory | Status |
|-----------|-------|-----------|--------|
| trellis | TRELLIS 1.0 (SLAT + Rectified Flow) | `models/trellis/` | Used in past experiments; not in new ones |
| hunyuan3d | Hunyuan3D 2.0 (DiT + Texture) | `models/hunyuan3d/` | Used in past experiments; not in new ones |
| sam3d | SAM-3D-Objects (Meta) | `models/sam3d/` | Legacy results and implementation reference. Excluded from pipeline |

Not evaluated: CAST (external API dependency), ShapeR (requires SLAM input)

## Directory Structure

```
.
├── docs/               # Architecture survey, experiment plan
├── models/             # Git repos for each model (2 active + 3 reference)
├── datasets/
│   └── Toys4k/
│       ├── official/   #   Unmodified distribution (blend, obj, point clouds, sample renders)
│       ├── renders/    #   Locally generated: <res>/<category>/<oid>/{image,depth,segmentation}.png
│       ├── zips/       #   Original zip archives
│       └── compat/     #   Legacy path symlinks for past experiments
├── pipeline/           # Evaluation infrastructure
│   ├── scripts/        #   Inference & evaluation scripts
│   ├── src/            #   Shared library (geometry, evaluation, data)
│   └── docs/           #   Evaluation protocol, model notes
├── experiments/        # Per-experiment config + run.sh + README
│   ├── _template/      #   Copy to create new experiments
│   ├── toys4k_baseline/
│   ├── resolution_sweep/
│   ├── category_pilot/
│   └── examples_qual/
├── results/            # Experiment outputs
│   └── toys4k/         #   Per-dataset
│       ├── predictions/           # Raw model outputs ({model}/default/{sample}/)
│       ├── normalized_predictions/ # Cleaned meshes / eval point clouds / alignment JSON
│       ├── metrics/               # Quantitative results CSV
│       ├── qualitative/           # Visualization grids
│       └── report.md              # Results report
├── archive/            # Retired experiments (smoke tests, v1 results)
└── envs/               # All runtime environments
    ├── miniconda3/     #   conda: trellis, trellis2
    ├── hunyuan3d-venv/ #   venv: hunyuan3d + hunyuan3d21
    ├── sam3d-mamba/    #   mamba: sam3d (reference only)
    └── cuda-12.8/      #   CUDA toolkit
```

## Quick Start

```bash
# Full pipeline (inference + evaluation + report)
bash experiments/toys4k_baseline/run.sh

# Evaluation only (skip inference, use existing predictions)
bash experiments/toys4k_baseline/run.sh --skip-inference

# Resolution sweep
bash experiments/resolution_sweep/run.sh

# View results
cat results/toys4k/report.md
cat results/toys4k/metrics/summary.csv
```

## Key Documents

| Document | Description |
|----------|-------------|
| [Architecture Survey](docs/architecture_survey.md) | Architecture comparison of candidate models |
| [Experiment Plan](docs/experiment_plan.md) | Initial small-scale comparison plan (historical) |
| [Evaluation Protocol](pipeline/docs/evaluation_protocol.md) | Current metric definitions & normalization procedures |
| [Model Notes](pipeline/docs/model_notes.md) | Model-specific notes & known issues |
| [Toys4k Plan](pipeline/docs/toys4k_plan.md) | Full-scale Toys4k evaluation plan (historical) |
| [Results Report](results/toys4k/report.md) | Latest evaluation results |

## Hardware

- 4x NVIDIA L20X 144GB (compute capability 8.9 / sm_89)
- System CUDA: 13.0 (`/usr/local/cuda-13.0/`)
- Build CUDA: 12.8 (`envs/cuda-12.8/`)

## Known Issues

- **CUDA 13.0/12.8 mismatch**: Custom kernels like nvdiffrast only work in the trellis env (CUDA 12.4). Fails with error 35 in hunyuan3d venv.
- **Hunyuan3D 2.1**: banana_028, screwdriver_016, plate_005 show scale anomalies (alignment scale 0.25-0.59).
