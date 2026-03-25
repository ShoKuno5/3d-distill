#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
EXP_DIR="$PROJECT_DIR/experiments/inference_profiling_seeds3"
BASE_CONFIG="$EXP_DIR/config.yaml"
OUTPUT_ROOT="$PROJECT_DIR/results/inference_profiling_seeds3"
TRELLIS2_PY="$PROJECT_DIR/envs/miniconda3/envs/trellis2/bin/python"
H3D_PY="$PROJECT_DIR/envs/hunyuan3d-venv/bin/python"
BLENDER="$PROJECT_DIR/envs/blender-3.6.16-linux-x64/blender"
RENDER_SCRIPT="$PROJECT_DIR/pipeline/scripts/_blender_render_mesh.py"

SEEDS=(0 1 2)

# ============================================================================
# Setup: splits + per-seed configs
# ============================================================================
SPLIT_DIR="$OUTPUT_ROOT/splits"
CONFIG_DIR="$OUTPUT_ROOT/configs"
mkdir -p "$SPLIT_DIR" "$CONFIG_DIR" "$OUTPUT_ROOT/logs"

python3 "$EXP_DIR/make_splits.py" \
    --manifest "$EXP_DIR/manifest.csv" \
    --output-dir "$SPLIT_DIR" \
    --n-splits 2

for SEED in "${SEEDS[@]}"; do
    python3 "$EXP_DIR/gen_seed_config.py" \
        --config "$BASE_CONFIG" \
        --seed "$SEED" \
        --output "$CONFIG_DIR/config_seed_${SEED}.yaml"
done

# ============================================================================
# Phase 1: Profiling + Mesh generation (3 rounds x 4 GPUs)
# ============================================================================
echo "=== Phase 1: Profiling + Mesh generation (seeds: ${SEEDS[*]}) ==="

for SEED in "${SEEDS[@]}"; do
    echo "--- Seed $SEED ---"
    SEED_CONFIG="$CONFIG_DIR/config_seed_${SEED}.yaml"

    # trellis2 on GPU 0,1
    cd "$PROJECT_DIR/models/trellis2"
    PYTHONPATH=. CUDA_HOME="$PROJECT_DIR/envs/cuda-12.8" \
      CUDA_VISIBLE_DEVICES=0 $TRELLIS2_PY \
      "$EXP_DIR/run_profile_trellis2.py" \
      --config "$SEED_CONFIG" --seed "$SEED" \
      --sample-ids "$SPLIT_DIR/split_0.txt" \
    > "$OUTPUT_ROOT/logs/profile_trellis2_seed${SEED}_gpu0.log" 2>&1 &

    PYTHONPATH=. CUDA_HOME="$PROJECT_DIR/envs/cuda-12.8" \
      CUDA_VISIBLE_DEVICES=1 $TRELLIS2_PY \
      "$EXP_DIR/run_profile_trellis2.py" \
      --config "$SEED_CONFIG" --seed "$SEED" \
      --sample-ids "$SPLIT_DIR/split_1.txt" \
    > "$OUTPUT_ROOT/logs/profile_trellis2_seed${SEED}_gpu1.log" 2>&1 &

    # hunyuan3d21 on GPU 2,3
    cd "$PROJECT_DIR/models/hunyuan3d21"
    PYTHONPATH=. LD_PRELOAD=/opt/unreal-engine/usr/lib/x86_64-linux-gnu/libOpenGL.so.0 \
      CUDA_VISIBLE_DEVICES=2 $H3D_PY \
      "$EXP_DIR/run_profile_hunyuan3d21.py" \
      --config "$SEED_CONFIG" --seed "$SEED" \
      --sample-ids "$SPLIT_DIR/split_0.txt" \
    > "$OUTPUT_ROOT/logs/profile_hunyuan3d21_seed${SEED}_gpu2.log" 2>&1 &

    PYTHONPATH=. LD_PRELOAD=/opt/unreal-engine/usr/lib/x86_64-linux-gnu/libOpenGL.so.0 \
      CUDA_VISIBLE_DEVICES=3 $H3D_PY \
      "$EXP_DIR/run_profile_hunyuan3d21.py" \
      --config "$SEED_CONFIG" --seed "$SEED" \
      --sample-ids "$SPLIT_DIR/split_1.txt" \
    > "$OUTPUT_ROOT/logs/profile_hunyuan3d21_seed${SEED}_gpu3.log" 2>&1 &

    echo "Seed $SEED: profiling launched on 4 GPUs. Waiting..."
    wait
    echo "Seed $SEED: profiling complete."
done

echo "Phase 1 complete."

# ============================================================================
# Phase 2: Evaluation (3 rounds x 16 CPU workers)
# ============================================================================
echo "=== Phase 2: Evaluation ==="
cd "$PROJECT_DIR"

for SEED in "${SEEDS[@]}"; do
    echo "--- Evaluating seed $SEED ---"
    SEED_CONFIG="$CONFIG_DIR/config_seed_${SEED}.yaml"
    $H3D_PY pipeline/scripts/run_eval.py \
      --config "$SEED_CONFIG" --workers 16 \
      2>&1 | tee "$OUTPUT_ROOT/logs/eval_seed_${SEED}.log"
done

echo "Phase 2 complete."

# ============================================================================
# Phase 3: Render predicted meshes (seed 0 only, 4 GPUs)
# ============================================================================
echo "=== Phase 3: Rendering predicted meshes (seed 0 only) ==="
RENDER_DIR="$OUTPUT_ROOT/renders"
mkdir -p "$RENDER_DIR"

JOBFILE="$OUTPUT_ROOT/logs/render_jobs.txt"
> "$JOBFILE"

GPU_IDX=0
NUM_GPUS=4
for MODEL_DIR in "$OUTPUT_ROOT/predictions"/*/seed_0; do
    MODEL_NAME="$(basename "$(dirname "$MODEL_DIR")")"
    mkdir -p "$RENDER_DIR/$MODEL_NAME"
    for OBJ_PATH in "$MODEL_DIR"/*/mesh_raw.obj; do
        [ -f "$OBJ_PATH" ] || continue
        OID="$(basename "$(dirname "$OBJ_PATH")")"
        OUT_PNG="$RENDER_DIR/$MODEL_NAME/${OID}.png"
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
# Phase 4: Per-seed timing analysis
# ============================================================================
echo "=== Phase 4: Per-seed timing analysis ==="
cd "$PROJECT_DIR"

for SEED in "${SEEDS[@]}"; do
    echo "--- Timing analysis seed $SEED ---"
    SEED_CONFIG="$CONFIG_DIR/config_seed_${SEED}.yaml"
    $H3D_PY "$EXP_DIR/analyze_timing.py" \
      --config "$SEED_CONFIG" \
      2>&1 | tee "$OUTPUT_ROOT/logs/timing_seed_${SEED}.log"
done

echo "Phase 4 complete."

# ============================================================================
# Phase 5: Cross-seed aggregation + report
# ============================================================================
echo "=== Phase 5: Cross-seed aggregation + report ==="
cd "$PROJECT_DIR"

EVAL_DIRS=()
for SEED in "${SEEDS[@]}"; do
    EVAL_DIRS+=("$OUTPUT_ROOT/eval_seed_${SEED}")
done

$H3D_PY "$EXP_DIR/analyze_seeds.py" \
    --config "$BASE_CONFIG" \
    --eval-dirs "${EVAL_DIRS[@]}" \
    --predictions-root "$OUTPUT_ROOT/predictions" \
    --seeds "${SEEDS[@]}" \
    --output-dir "$OUTPUT_ROOT/aggregated" \
    2>&1 | tee "$OUTPUT_ROOT/logs/analyze_seeds.log"

# Generate standard report from seed 0
$H3D_PY pipeline/scripts/generate_report.py \
    --config "$CONFIG_DIR/config_seed_0.yaml" \
    2>&1 | tee "$OUTPUT_ROOT/logs/report.log"

# Move seed_0 report to final location
if [ -f "$OUTPUT_ROOT/eval_seed_0/eval_report.md" ]; then
    cp "$OUTPUT_ROOT/eval_seed_0/eval_report.md" "$OUTPUT_ROOT/eval_report_seed0.md"
fi

echo "Done. Results at $OUTPUT_ROOT/"
echo "  Aggregated cross-seed analysis: $OUTPUT_ROOT/aggregated/"
echo "  Per-seed eval: $OUTPUT_ROOT/eval_seed_{0,1,2}/"
echo "  Renders (seed 0): $OUTPUT_ROOT/renders/"
