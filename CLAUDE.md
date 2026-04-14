# CLAUDE.md

3D generation model evaluation & distillation workspace. Comparison pipeline + 5 model repos.

Key docs: `docs/architecture_survey.md`, `docs/experiment_plan.md`

## Directory Structure

```
distillation/
├── models/             # Git repos: trellis, trellis2, hunyuan3d, hunyuan3d21, sam3d
├── datasets/
│   └── Toys4k/
│       ├── official/   # Unmodified files from official distribution (blend, obj, point clouds, sample renders)
│       ├── renders/    # Locally generated renders: renders/<res>/<category>/<oid>/{image,depth,segmentation}.png
│       ├── zips/       # Original zip archives
│       └── compat/     # Symlinks preserving legacy paths for past experiments
├── pipeline/           # Evaluation infrastructure: scripts/, src/, docs/
├── distill_methods/    # Distillation experiment (branch: exp/distill-methods-comparison)
│   ├── config.yaml     # Full experiment config (dataset, training, models, metrics)
│   ├── manifest.csv    # Sample list for this experiment
│   ├── src/            # Training code: base_distiller, pd/cd/dmd1/dmd2, train.py
│   ├── scripts/        # Data prep, inference, comparison scripts
│   └── reports/        # Experiment reports (generated via /report-experiment)
├── results/            # Predictions, metrics, reports
├── envs/               # Runtime environments (not shared across repos)
├── docs/               # Project-wide documents
└── archive/            # Past experiments (includes archive/experiments/)
```

## Past Experiments (archived)

Past experiments have been moved to `archive/experiments/`. They are records and should not be edited.

```bash
bash archive/experiments/toys4k_baseline/run.sh          # 4-model baseline comparison
bash archive/experiments/resolution_sweep/run.sh         # Input resolution sweep
bash archive/experiments/examples_qual/run.sh            # Qualitative output examples
```

Note: `run.sh` scripts use `$SCRIPT_DIR`-relative path resolution. After the move, `PROJECT_DIR` may not resolve correctly — fix the depth if re-running.

Common options: `--skip-inference`, `--workers=N`, `--max-samples N`

### Branch Convention

- `main` — stable shared infrastructure (pipeline, docs, dataset layout)
- `exp/<name>` — one branch per experiment (e.g. `exp/category-pilot`, `exp/resolution-sweep`)

## Running Each Model (standalone)

```bash
cd models/hunyuan3d   && ../../envs/hunyuan3d-venv/bin/python minimal_demo.py
cd models/hunyuan3d21 && PYTHONPATH=. ../../envs/hunyuan3d-venv/bin/python demo.py
cd models/trellis     && PYTHONPATH=. ../../envs/miniconda3/envs/trellis/bin/python example.py
cd models/trellis2    && PYTHONPATH=. ../../envs/miniconda3/envs/trellis2/bin/python example.py
cd models/sam3d       && CONDA_PREFIX=../../envs/sam3d-mamba/envs/sam3d-objects \
  ../../envs/sam3d-mamba/envs/sam3d-objects/bin/python demo.py
```

## Running Distillation (distill_methods/)

CWD must be `models/hunyuan3d21/hy3dshape` for hy3dshape imports to resolve.

```bash
cd models/hunyuan3d21/hy3dshape

# Training (single GPU)
PYTHONPATH=.:../../../distill_methods/src CUDA_VISIBLE_DEVICES=0 \
  ../../../envs/hunyuan3d-venv/bin/python -u ../../../distill_methods/src/train.py \
  --config ../../../distill_methods/config.yaml --method pd --stage 0

# Training (multi-GPU via DDP)
PYTHONPATH=.:../../../distill_methods/src CUDA_VISIBLE_DEVICES=0,1,2,3 \
  torchrun --nproc_per_node=4 ../../../distill_methods/src/train.py \
  --config ../../../distill_methods/config.yaml --method pd --stage 0
```

## Hardware

- **GPUs**: 4x NVIDIA L20X 144GB (compute capability 8.9, sm_89)
- **Storage**: Alibaba CPFS (shared network filesystem). Initial model loads are slow (~5 min for 6.9GB ckpt); subsequent loads use OS page cache.

## Rendering

- **Blender 3.6** with GPU rendering (CYCLES + CUDA). Do not use CPU device.

## Rules

- Never fabricate weights URLs or dataset links — only cite what exists in the repo.
- If something is unclear, write "Unknown" rather than guessing.
- Do not share conda/venv environments between repos (version conflicts).
- **Do not edit past experiments**: Archived experiments in `archive/experiments/` are records. Create a new experiment directory to change conditions. (Path-only updates for dataset reorganization are allowed.)
- **Dataset paths**: New experiments reference `datasets/Toys4k/official/` and `datasets/Toys4k/renders/` directly. Past experiments use `datasets/Toys4k/compat/` symlinks.
- **Active models**: New experiments use trellis2 and hunyuan3d21 only. Others may be re-added later.
- **Comments in English**: All comments, docstrings, and documentation should be written in English.
- **Geometry only**: Focus on structure and geometry. Zero interest in appearance/texture.
- **No hardcoded hyperparameters**: All training hyperparameters (lr, weight_decay, total_steps, etc.) must come from `config.yaml`. Shared params go in `training.optimizer` or top-level `training.*`; method-specific params go in `training.methods.<name>`. Never hardcode numeric values in Python code.
- **Experiment reports**: Use `/report-experiment [focus]` to generate reports. Reports are saved to `distill_methods/reports/` as markdown. Uses git diff to detect new/changed results since the last `[report]` commit. Reports are agentic — Claude reads all logs, config, and code to write analysis, not mechanical parsing.
