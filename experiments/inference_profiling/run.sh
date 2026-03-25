#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG="$PROJECT_DIR/experiments/inference_profiling/config.yaml"
OUTPUT_ROOT="$PROJECT_DIR/results/inference_profiling"
TRELLIS2_PY="$PROJECT_DIR/envs/miniconda3/envs/trellis2/bin/python"
H3D_PY="$PROJECT_DIR/envs/hunyuan3d-venv/bin/python"

# Split sample IDs for 2-way parallel per model
SPLIT_DIR="$OUTPUT_ROOT/splits"
mkdir -p "$SPLIT_DIR" "$OUTPUT_ROOT/logs"

python3 "$PROJECT_DIR/experiments/inference_profiling/make_splits.py" \
    --manifest "$PROJECT_DIR/experiments/inference_profiling/manifest.csv" \
    --output-dir "$SPLIT_DIR" \
    --n-splits 2

# Inference: 2 models x 2 splits = 4 GPUs
# Each process warms up with its first manifest sample before recording.

echo "=== Launching profiling on 4 GPUs ==="

# trellis2 on GPU 0,1
cd "$PROJECT_DIR/models/trellis2"
PYTHONPATH=. CUDA_HOME="$PROJECT_DIR/envs/cuda-12.8" \
  CUDA_VISIBLE_DEVICES=0 $TRELLIS2_PY \
  "$PROJECT_DIR/experiments/inference_profiling/run_profile_trellis2.py" \
  --config "$CONFIG" --sample-ids "$SPLIT_DIR/split_0.txt" \
> "$OUTPUT_ROOT/logs/profile_trellis2_gpu0.log" 2>&1 &

PYTHONPATH=. CUDA_HOME="$PROJECT_DIR/envs/cuda-12.8" \
  CUDA_VISIBLE_DEVICES=1 $TRELLIS2_PY \
  "$PROJECT_DIR/experiments/inference_profiling/run_profile_trellis2.py" \
  --config "$CONFIG" --sample-ids "$SPLIT_DIR/split_1.txt" \
> "$OUTPUT_ROOT/logs/profile_trellis2_gpu1.log" 2>&1 &

# hunyuan3d21 on GPU 2,3
cd "$PROJECT_DIR/models/hunyuan3d21"
PYTHONPATH=. LD_PRELOAD=/opt/unreal-engine/usr/lib/x86_64-linux-gnu/libOpenGL.so.0 \
  CUDA_VISIBLE_DEVICES=2 $H3D_PY \
  "$PROJECT_DIR/experiments/inference_profiling/run_profile_hunyuan3d21.py" \
  --config "$CONFIG" --sample-ids "$SPLIT_DIR/split_0.txt" \
> "$OUTPUT_ROOT/logs/profile_hunyuan3d21_gpu2.log" 2>&1 &

PYTHONPATH=. LD_PRELOAD=/opt/unreal-engine/usr/lib/x86_64-linux-gnu/libOpenGL.so.0 \
  CUDA_VISIBLE_DEVICES=3 $H3D_PY \
  "$PROJECT_DIR/experiments/inference_profiling/run_profile_hunyuan3d21.py" \
  --config "$CONFIG" --sample-ids "$SPLIT_DIR/split_1.txt" \
> "$OUTPUT_ROOT/logs/profile_hunyuan3d21_gpu3.log" 2>&1 &

echo "Profiling launched on 4 GPUs. Waiting..."
wait
echo "Profiling complete."

# Analyze results
echo "=== Analyzing results ==="
cd "$PROJECT_DIR"
$H3D_PY "$PROJECT_DIR/experiments/inference_profiling/analyze_timing.py" \
  --config "$CONFIG" \
  2>&1 | tee "$OUTPUT_ROOT/logs/analyze.log"

echo "Done. Results at $OUTPUT_ROOT/"
