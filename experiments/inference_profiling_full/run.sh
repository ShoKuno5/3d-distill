#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
EXP_DIR="$PROJECT_DIR/experiments/inference_profiling_full"
CONFIG="$EXP_DIR/config.yaml"
OUTPUT_ROOT="$PROJECT_DIR/results/inference_profiling_full"
TRELLIS2_PY="$PROJECT_DIR/envs/miniconda3/envs/trellis2/bin/python"
H3D_PY="$PROJECT_DIR/envs/hunyuan3d-venv/bin/python"
BLENDER="$PROJECT_DIR/envs/blender-3.6.16-linux-x64/blender"
RENDER_SCRIPT="$PROJECT_DIR/pipeline/scripts/_blender_render_mesh.py"

# Split sample IDs for 2-way parallel per model
SPLIT_DIR="$OUTPUT_ROOT/splits"
mkdir -p "$SPLIT_DIR" "$OUTPUT_ROOT/logs"

python3 "$EXP_DIR/make_splits.py" \
    --manifest "$EXP_DIR/manifest.csv" \
    --output-dir "$SPLIT_DIR" \
    --n-splits 2

# ============================================================================
# Phase 1: Profiling + Mesh generation (4 GPUs, parallel)
# ============================================================================
echo "=== Phase 1: Profiling + Mesh generation on 4 GPUs ==="

# trellis2 on GPU 0,1
cd "$PROJECT_DIR/models/trellis2"
PYTHONPATH=. CUDA_HOME="$PROJECT_DIR/envs/cuda-12.8" \
  CUDA_VISIBLE_DEVICES=0 $TRELLIS2_PY \
  "$EXP_DIR/run_profile_trellis2.py" \
  --config "$CONFIG" --sample-ids "$SPLIT_DIR/split_0.txt" \
> "$OUTPUT_ROOT/logs/profile_trellis2_gpu0.log" 2>&1 &

PYTHONPATH=. CUDA_HOME="$PROJECT_DIR/envs/cuda-12.8" \
  CUDA_VISIBLE_DEVICES=1 $TRELLIS2_PY \
  "$EXP_DIR/run_profile_trellis2.py" \
  --config "$CONFIG" --sample-ids "$SPLIT_DIR/split_1.txt" \
> "$OUTPUT_ROOT/logs/profile_trellis2_gpu1.log" 2>&1 &

# hunyuan3d21 on GPU 2,3
cd "$PROJECT_DIR/models/hunyuan3d21"
PYTHONPATH=. LD_PRELOAD=/opt/unreal-engine/usr/lib/x86_64-linux-gnu/libOpenGL.so.0 \
  CUDA_VISIBLE_DEVICES=2 $H3D_PY \
  "$EXP_DIR/run_profile_hunyuan3d21.py" \
  --config "$CONFIG" --sample-ids "$SPLIT_DIR/split_0.txt" \
> "$OUTPUT_ROOT/logs/profile_hunyuan3d21_gpu2.log" 2>&1 &

PYTHONPATH=. LD_PRELOAD=/opt/unreal-engine/usr/lib/x86_64-linux-gnu/libOpenGL.so.0 \
  CUDA_VISIBLE_DEVICES=3 $H3D_PY \
  "$EXP_DIR/run_profile_hunyuan3d21.py" \
  --config "$CONFIG" --sample-ids "$SPLIT_DIR/split_1.txt" \
> "$OUTPUT_ROOT/logs/profile_hunyuan3d21_gpu3.log" 2>&1 &

echo "Profiling launched on 4 GPUs. Waiting..."
wait
echo "Phase 1 complete."

# ============================================================================
# Phase 2: Evaluation (CPU-only, parallel via --workers)
# ============================================================================
echo "=== Phase 2: Evaluation ==="
cd "$PROJECT_DIR"
$H3D_PY pipeline/scripts/run_eval.py \
  --config "$CONFIG" --workers 16 \
  2>&1 | tee "$OUTPUT_ROOT/logs/eval.log"

echo "Phase 2 complete."

# ============================================================================
# Phase 3: Render predicted meshes (4 GPUs, parallel via xargs)
# ============================================================================
echo "=== Phase 3: Rendering predicted meshes ==="
RENDER_DIR="$OUTPUT_ROOT/renders"
mkdir -p "$RENDER_DIR"

# Build job list: for each model, find all predicted OBJ files
JOBFILE="$OUTPUT_ROOT/logs/render_jobs.txt"
> "$JOBFILE"

GPU_IDX=0
NUM_GPUS=4
for MODEL_DIR in "$OUTPUT_ROOT/predictions"/*/default; do
    MODEL_NAME="$(basename "$(dirname "$MODEL_DIR")")"
    mkdir -p "$RENDER_DIR/$MODEL_NAME"
    for OBJ_PATH in "$MODEL_DIR"/*/mesh_raw.obj; do
        [ -f "$OBJ_PATH" ] || continue
        OID="$(basename "$(dirname "$OBJ_PATH")")"
        OUT_PNG="$RENDER_DIR/$MODEL_NAME/${OID}.png"
        # Round-robin GPU assignment
        GPU=$((GPU_IDX % NUM_GPUS))
        GPU_IDX=$((GPU_IDX + 1))
        echo "CUDA_VISIBLE_DEVICES=$GPU $BLENDER --background --python $RENDER_SCRIPT -- --input $OBJ_PATH --output $OUT_PNG --resolution 512" >> "$JOBFILE"
    done
done

N_JOBS=$(wc -l < "$JOBFILE")
echo "Rendering $N_JOBS meshes across $NUM_GPUS GPUs..."

if [ "$N_JOBS" -gt 0 ]; then
    xargs -P "$NUM_GPUS" -I {} bash -c '{}' < "$JOBFILE" \
      > "$OUTPUT_ROOT/logs/render.log" 2>&1
fi

echo "Phase 3 complete."

# ============================================================================
# Phase 4: Analysis + Report
# ============================================================================
echo "=== Phase 4: Analysis + Report ==="
cd "$PROJECT_DIR"

$H3D_PY "$EXP_DIR/analyze_timing.py" \
  --config "$CONFIG" \
  2>&1 | tee "$OUTPUT_ROOT/logs/analyze.log"

$H3D_PY pipeline/scripts/generate_report.py \
  --config "$CONFIG" \
  2>&1 | tee "$OUTPUT_ROOT/logs/report.log"

echo "Done. Results at $OUTPUT_ROOT/"
