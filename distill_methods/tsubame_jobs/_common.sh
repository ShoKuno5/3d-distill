#!/usr/bin/env bash
# Shared setup for TSUBAME 4.0 distill jobs.
#
# Layout assumptions:
#   TSUBAME_ROOT=/gs/fs/tga-koike-shanda2/sk
#     ├── 3d-distill/              (repo clone, this branch: exp/scaling-experiments)
#     │   └── .venv/               (uv venv, Python 3.10)
#     ├── hf_cache/                (HuggingFace cache; tencent/Hunyuan3D-2.1)
#     ├── data/                    (pre-encoded latents + DMD1 pairs + toys4k renders)
#     │   ├── 525_hssd/training_data/*.npz
#     │   ├── 525_hssd/dmd1_pairs/pair_*.npz
#     │   └── toys4k/renders/...
#     └── scratch/distill_methods/ (per-run output)
#
# Usage from a job script:
#   source "$(dirname "$0")/_common.sh"
#   tsubame_setup_env
#   tsubame_log_init m1_pd

set -euo pipefail

TSUBAME_ROOT="${TSUBAME_ROOT:-/gs/fs/tga-koike-shanda2/sk}"
REPO="$TSUBAME_ROOT/3d-distill"
VENV="$REPO/.venv"
PY="$VENV/bin/python"
TORCHRUN="$VENV/bin/torchrun"

tsubame_setup_env() {
    module purge 2>/dev/null || true
    module load cuda/12.8.0 cudnn/9.8.0 2>/dev/null || echo "[warn] module load failed (interactive shell?)"

    export HF_HOME="$TSUBAME_ROOT/hf_cache"
    export PYTHONPATH="$REPO/models/hunyuan3d21/hy3dshape:$REPO/distill_methods/src${PYTHONPATH:+:$PYTHONPATH}"
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    # WANDB online OK on TSUBAME (no egress restriction unlike qzcli)
    if [ -f "$REPO/.env" ]; then
        set -a; . "$REPO/.env"; set +a
    fi
    : "${WANDB_API_KEY:?WANDB_API_KEY must be set (put it in $REPO/.env)}"
}

tsubame_log_init() {
    local NAME="$1"
    local LOGDIR="$TSUBAME_ROOT/scratch/tsubame_logs"
    mkdir -p "$LOGDIR"
    local LOG="$LOGDIR/${NAME}_$(hostname)_$(date +%Y%m%d_%H%M%S).log"
    exec > >(tee -a "$LOG") 2>&1
    echo "[log mirror: $LOG]"
}

tsubame_print_banner() {
    local NAME="$1"
    local RUN_NAME="$2"
    local OUTPUT_ROOT="$3"
    local NGPU
    NGPU=$(nvidia-smi -L | wc -l)
    local GPU_NAME
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    local GPU_MEM
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    echo "================================================================"
    echo "$NAME: $RUN_NAME"
    echo "output: $OUTPUT_ROOT"
    echo "host: $(hostname)  ngpu: $NGPU  gpu: $GPU_NAME ($GPU_MEM)"
    echo "================================================================"
}

# Build a per-run config copy (output_root rewritten, optional overrides).
# Usage: tsubame_make_run_config <RUN_CFG> [extra python override line]
tsubame_make_run_config() {
    local OUTPUT_ROOT="$1"
    local RUN_CFG="$2"
    local EXTRA="${3:-}"
    local SRC_CFG="$REPO/distill_methods/configs/config_525_hssd_tsubame.yaml"
    mkdir -p "$OUTPUT_ROOT"
    if [ -f "$RUN_CFG" ]; then
        echo "config exists: $RUN_CFG (reusing)"
        return
    fi
    "$PY" - <<PY
import yaml
cfg = yaml.safe_load(open('$SRC_CFG'))
cfg['output_root'] = '$OUTPUT_ROOT'
cfg['training']['batch_size'] = 4
$EXTRA
yaml.safe_dump(cfg, open('$RUN_CFG', 'w'), default_flow_style=False, sort_keys=False)
print('config:', '$RUN_CFG')
PY
}
