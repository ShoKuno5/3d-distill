#!/usr/bin/env bash
# Run the complete 4-model evaluation pipeline.
#
# Usage:
#   bash pipeline/scripts/run_all_eval.sh [--config=PATH] [--max-samples N] [--skip-inference] [--workers N]
#
# --config defaults to experiments/toys4k_baseline/config.yaml if not provided.
#
# Inference runs fully parallel: 1 model per GPU (4 GPUs).
#   GPU 0: trellis
#   GPU 1: hunyuan3d
#   GPU 2: trellis2
#   GPU 3: hunyuan3d21

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
CONFIG=""

# Parse args
MAX_SAMPLES_ARG=""
SKIP_INFERENCE=false
WORKERS_ARG=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --config=*) CONFIG="${1#*=}"; shift ;;
        --config) CONFIG="$2"; shift 2 ;;
        --max-samples=*) MAX_SAMPLES_ARG="--max-samples ${1#*=}"; shift ;;
        --max-samples) MAX_SAMPLES_ARG="--max-samples $2"; shift 2 ;;
        --skip-inference) SKIP_INFERENCE=true; shift ;;
        --workers=*) WORKERS_ARG="--workers ${1#*=}"; shift ;;
        --workers) WORKERS_ARG="--workers $2"; shift 2 ;;
        *) echo "WARNING: Unknown argument: $1"; shift ;;
    esac
done

# Default config if not specified
if [ -z "$CONFIG" ]; then
    CONFIG="$PROJECT_DIR/experiments/toys4k_baseline/config.yaml"
fi

# Resolve to absolute path if relative
if [[ "$CONFIG" != /* ]]; then
    CONFIG="$PROJECT_DIR/$CONFIG"
fi

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config file not found: $CONFIG"
    exit 1
fi

echo "Using config: $CONFIG"

# Python interpreters
H3D_PYTHON="$PROJECT_DIR/envs/hunyuan3d-venv/bin/python"
TRELLIS_PYTHON="$PROJECT_DIR/envs/miniconda3/envs/trellis/bin/python"
TRELLIS2_PYTHON="$PROJECT_DIR/envs/miniconda3/envs/trellis2/bin/python"

# Eval python (uses H3D venv which has trimesh/scipy/numpy)
EVAL_PYTHON="$H3D_PYTHON"

# Extract output_root from config for log directory
OUTPUT_ROOT=$($H3D_PYTHON -c "import yaml,sys; cfg=yaml.safe_load(open('$CONFIG')); print(cfg.get('output_root','$PROJECT_DIR/results/toys4k'))")
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR"

if [ "$SKIP_INFERENCE" = false ]; then
    echo "========================================="
    echo "Inference (4 models × 4 GPUs, fully parallel)"
    echo "========================================="

    # GPU 0: trellis
    (
        cd "$PROJECT_DIR/models/trellis"
        PYTHONPATH=. SPCONV_ALGO=native CUDA_VISIBLE_DEVICES=0 \
        $TRELLIS_PYTHON ../../pipeline/scripts/run_inference_trellis.py \
            --config "$CONFIG" $MAX_SAMPLES_ARG \
            2>&1 | tee "$LOG_DIR/inference_trellis.log"
    ) &
    PID_GPU0=$!

    # GPU 1: hunyuan3d
    (
        cd "$PROJECT_DIR/models/hunyuan3d"
        PYTHONPATH=. \
        LD_PRELOAD=/opt/unreal-engine/usr/lib/x86_64-linux-gnu/libOpenGL.so.0 \
        CUDA_VISIBLE_DEVICES=1 \
        $H3D_PYTHON ../../pipeline/scripts/run_inference_hunyuan3d2.py \
            --config "$CONFIG" $MAX_SAMPLES_ARG \
            2>&1 | tee "$LOG_DIR/inference_hunyuan3d.log"
    ) &
    PID_GPU1=$!

    # GPU 2: trellis2
    (
        cd "$PROJECT_DIR/models/trellis2"
        PYTHONPATH=. CUDA_HOME="$PROJECT_DIR/envs/cuda-12.8" CUDA_VISIBLE_DEVICES=2 \
        $TRELLIS2_PYTHON ../../pipeline/scripts/run_inference_trellis2.py \
            --config "$CONFIG" $MAX_SAMPLES_ARG \
            2>&1 | tee "$LOG_DIR/inference_trellis2.log"
    ) &
    PID_GPU2=$!

    # GPU 3: hunyuan3d21
    (
        cd "$PROJECT_DIR/models/hunyuan3d21"
        PYTHONPATH=. \
        LD_PRELOAD=/opt/unreal-engine/usr/lib/x86_64-linux-gnu/libOpenGL.so.0 \
        CUDA_VISIBLE_DEVICES=3 \
        $H3D_PYTHON ../../pipeline/scripts/run_inference_hunyuan3d21.py \
            --config "$CONFIG" $MAX_SAMPLES_ARG \
            2>&1 | tee "$LOG_DIR/inference_hunyuan3d21.log"
    ) &
    PID_GPU3=$!

    echo "  GPU0 (trellis):    PID $PID_GPU0"
    echo "  GPU1 (hunyuan3d):  PID $PID_GPU1"
    echo "  GPU2 (trellis2):   PID $PID_GPU2"
    echo "  GPU3 (hunyuan3d21): PID $PID_GPU3"
    echo "  Logs: $LOG_DIR/inference_*.log"
    echo ""

    # Wait for all
    FAILED=0
    wait $PID_GPU0 || { echo "ERROR: trellis inference failed"; FAILED=1; }
    wait $PID_GPU1 || { echo "ERROR: hunyuan3d inference failed"; FAILED=1; }
    wait $PID_GPU2 || { echo "ERROR: trellis2 inference failed"; FAILED=1; }
    wait $PID_GPU3 || { echo "ERROR: hunyuan3d21 inference failed"; FAILED=1; }

    if [ $FAILED -ne 0 ]; then
        echo "WARNING: Some inference jobs failed. Continuing with available predictions..."
    fi
    echo ""
    echo "All inference jobs finished."
fi

echo ""
echo "========================================="
echo "Check inference status"
echo "========================================="
cd "$PROJECT_DIR"
$EVAL_PYTHON pipeline/scripts/run_all_inference.py --config "$CONFIG" $MAX_SAMPLES_ARG

echo ""
echo "========================================="
echo "Evaluation (parallel)"
echo "========================================="
cd "$PROJECT_DIR"
$EVAL_PYTHON pipeline/scripts/run_eval.py --config "$CONFIG" $MAX_SAMPLES_ARG $WORKERS_ARG

echo ""
echo "========================================="
echo "Generate report"
echo "========================================="
$EVAL_PYTHON pipeline/scripts/generate_report.py --config "$CONFIG"

echo ""
echo "Done! Report: $OUTPUT_ROOT/report.md"
