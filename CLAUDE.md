# CLAUDE.md

3D generation model evaluation & distillation workspace. Comparison pipeline + 5 model repos.

Key docs: `docs/architecture_survey.md`, `docs/experiment_plan.md`

## Directory Structure

```
distillation/
├── models/             # Git repos: trellis, trellis2, hunyuan3d, hunyuan3d21, sam3d
├── datasets/           # Toys4k input images + GT point clouds
├── pipeline/           # Evaluation infrastructure: scripts/, src/, docs/
├── experiments/        # Per-experiment config + run.sh + README (copy _template/ to create new)
├── results/            # Predictions, metrics, reports
├── envs/               # Runtime environments (not shared across repos)
├── docs/               # Project-wide documents
└── archive/            # Past experiments
```

## Running Experiments

Each experiment runs via `experiments/<name>/run.sh`:

```bash
bash experiments/toys4k_baseline/run.sh                  # 4-model baseline comparison
bash experiments/resolution_sweep/run.sh                 # Input resolution sweep
bash experiments/examples_qual/run.sh                    # Qualitative output examples
```

Common options: `--skip-inference`, `--workers=N`, `--max-samples N`

## Running Each Model (standalone)

```bash
cd models/hunyuan3d   && ../../envs/hunyuan3d-venv/bin/python minimal_demo.py
cd models/hunyuan3d21 && PYTHONPATH=. ../../envs/hunyuan3d-venv/bin/python demo.py
cd models/trellis     && PYTHONPATH=. ../../envs/miniconda3/envs/trellis/bin/python example.py
cd models/trellis2    && PYTHONPATH=. ../../envs/miniconda3/envs/trellis2/bin/python example.py
cd models/sam3d       && CONDA_PREFIX=../../envs/sam3d-mamba/envs/sam3d-objects \
  ../../envs/sam3d-mamba/envs/sam3d-objects/bin/python demo.py
```

## Rules

- Never fabricate weights URLs or dataset links — only cite what exists in the repo.
- If something is unclear, write "Unknown" rather than guessing.
- Do not share conda/venv environments between repos (version conflicts).
- **Do not edit past experiments**: Existing experiments in `experiments/` (configs, READMEs) are records. Create a new experiment directory to change conditions.
- **Active models**: New experiments use trellis2 and hunyuan3d21 only. Others may be re-added later.
- **Comments in English**: All comments, docstrings, and documentation should be written in English.
