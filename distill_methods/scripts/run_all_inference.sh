#!/bin/bash
# Run inference for all distilled models across 4 GPUs.
# Usage: bash distill_methods/scripts/run_all_inference.sh [run_dir]

set -euo pipefail

PROJ=/mnt/workspace/kuno/distillation
RUN_DIR="${1:-$PROJ/results/distill_methods/runs/20260406_1131}"
# Ensure absolute path
RUN_DIR="$(cd "$(dirname "$RUN_DIR")" && pwd)/$(basename "$RUN_DIR")"
CONFIG=$PROJ/distill_methods/config.yaml
PYTHON=$PROJ/envs/hunyuan3d-venv/bin/python
CWD=$PROJ/models/hunyuan3d21/hy3dshape
PPATH=".:$PROJ/distill_methods/src:$PROJ/pipeline"

echo "Run dir: $RUN_DIR"
echo "Config:  $CONFIG"
echo ""

run_model() {
    local gpu=$1 model=$2
    echo "[GPU $gpu] Starting $model ..."
    cd "$CWD"
    PYTHONPATH=$PPATH CUDA_VISIBLE_DEVICES=$gpu \
        $PYTHON -u "$PROJ/distill_methods/scripts/run_inference_distilled.py" \
        --config "$CONFIG" --model-name "$model" --run-dir "$RUN_DIR" \
        2>&1 | sed "s/^/[GPU $gpu $model] /"
    echo "[GPU $gpu] $model done."
}

# Wave 1: 4 models on 4 GPUs
run_model 0 pd_6step &
run_model 1 cd_4step &
run_model 2 dmd1_1step &
run_model 3 dmd2_1step &
wait

# Wave 2: remaining model
run_model 0 sid_1step
wait

echo ""
echo "All inference complete."
