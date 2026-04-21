#!/usr/bin/env bash
# Cross-family Toys4k inference-evaluation benchmark orchestrator.
#
# 9 models × 105 test samples, wave-based GPU scheduling.
#
# Usage:
#   bash distill_methods/scripts/run_cross_family_benchmark.sh                       # full run
#   bash distill_methods/scripts/run_cross_family_benchmark.sh --max-samples 5       # quick test
#   bash distill_methods/scripts/run_cross_family_benchmark.sh --run-id RUN_ID       # resume
#   bash distill_methods/scripts/run_cross_family_benchmark.sh --skip-inference      # eval only

set -eu

# ---------- Constants ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
H3D_PY="$PROJECT_DIR/envs/hunyuan3d-venv/bin/python"
TRELLIS_PY="$PROJECT_DIR/envs/miniconda3/envs/trellis/bin/python"
TRELLIS2_PY="$PROJECT_DIR/envs/miniconda3/envs/trellis2/bin/python"
EVAL_PY="$H3D_PY"   # run_eval.py can use any env with yaml/trimesh/scipy
LD_PRELOAD_GL="/opt/unreal-engine/usr/lib/x86_64-linux-gnu/libOpenGL.so.0"
CONFIG_SRC="$PROJECT_DIR/distill_methods/config_cross_family.yaml"

# ---------- Args ----------
MAX_SAMPLES_ARG=""
RUN_ID=""
SKIP_INFERENCE=false
SKIP_EVAL=false
SKIP_SENSITIVITY=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-samples) MAX_SAMPLES_ARG="--max-samples $2"; shift 2 ;;
        --run-id) RUN_ID="$2"; shift 2 ;;
        --skip-inference) SKIP_INFERENCE=true; shift ;;
        --skip-eval) SKIP_EVAL=true; shift ;;
        --skip-sensitivity) SKIP_SENSITIVITY=true; shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$RUN_ID" ]]; then
    RUN_ID="$(date +%Y%m%d_%H%M)_cross_family"
fi
RUN_DIR="$PROJECT_DIR/results/distill_methods/runs/$RUN_ID"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$RUN_DIR" "$LOG_DIR"

echo "=========================================="
echo "Cross-family benchmark"
echo "  RUN_ID:   $RUN_ID"
echo "  RUN_DIR:  $RUN_DIR"
echo "  options:  skip_inference=$SKIP_INFERENCE skip_eval=$SKIP_EVAL skip_sensitivity=$SKIP_SENSITIVITY $MAX_SAMPLES_ARG"
echo "=========================================="

# ---------- Generate per-run config ----------
INFERENCE_CONFIG="$RUN_DIR/config_inference.yaml"
"$EVAL_PY" -c "
import yaml
with open('$CONFIG_SRC') as f: cfg = yaml.safe_load(f)
cfg['output_root'] = '$RUN_DIR'
# Resolve per-model predictions_root so later eval can override output_root for sensitivity runs
for m in cfg['models']:
    m['predictions_root'] = cfg['output_root'] + '/predictions/' + m['name'] + '/default'
with open('$INFERENCE_CONFIG', 'w') as f: yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
print('Wrote', '$INFERENCE_CONFIG')
"

# ---------- Phase 2: Inference (waves) ----------
if [[ "$SKIP_INFERENCE" == "false" ]]; then
    echo ""
    echo "===== Wave 1: Teachers (GPU 0-3 parallel) ====="

    # GPU0: H3D-2.1 teacher_50step
    ( cd "$PROJECT_DIR/models/hunyuan3d21/hy3dshape" && \
      PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 "$H3D_PY" \
        "$PROJECT_DIR/distill_methods/scripts/run_inference_distilled.py" \
        --config "$INFERENCE_CONFIG" --model-name teacher_50step $MAX_SAMPLES_ARG \
    ) > "$LOG_DIR/inference_teacher_50step.log" 2>&1 &
    P0=$!

    # GPU1: FlashVDM (v2.0 CWD)
    ( cd "$PROJECT_DIR/models/hunyuan3d" && \
      PYTHONPATH=. LD_PRELOAD="$LD_PRELOAD_GL" CUDA_VISIBLE_DEVICES=1 "$H3D_PY" \
        "$PROJECT_DIR/pipeline/scripts/run_inference_flashvdm.py" \
        --config "$INFERENCE_CONFIG" $MAX_SAMPLES_ARG \
    ) > "$LOG_DIR/inference_flashvdm.log" 2>&1 &
    P1=$!

    # GPU2: TRELLIS v1 teacher
    ( cd "$PROJECT_DIR/models/trellis" && \
      SPCONV_ALGO=native PYTHONPATH=. CUDA_VISIBLE_DEVICES=2 "$TRELLIS_PY" \
        "$PROJECT_DIR/pipeline/scripts/run_inference_trellis.py" \
        --config "$INFERENCE_CONFIG" $MAX_SAMPLES_ARG \
    ) > "$LOG_DIR/inference_trellis.log" 2>&1 &
    P2=$!

    # GPU3: TRELLIS 2 teacher
    ( cd "$PROJECT_DIR/models/trellis2" && \
      PYTHONPATH=. CUDA_HOME=/usr/local/cuda-12.4 CUDA_VISIBLE_DEVICES=3 "$TRELLIS2_PY" \
        "$PROJECT_DIR/pipeline/scripts/run_inference_trellis2.py" \
        --config "$INFERENCE_CONFIG" $MAX_SAMPLES_ARG \
    ) > "$LOG_DIR/inference_trellis2.log" 2>&1 &
    P3=$!

    echo "  GPU0 teacher_50step PID=$P0"
    echo "  GPU1 flashvdm       PID=$P1"
    echo "  GPU2 trellis        PID=$P2"
    echo "  GPU3 trellis2       PID=$P3"
    W1_FAIL=0
    wait $P0 || { echo "  ERROR teacher_50step"; W1_FAIL=1; }
    wait $P1 || { echo "  ERROR flashvdm";       W1_FAIL=1; }
    wait $P2 || { echo "  ERROR trellis";        W1_FAIL=1; }
    wait $P3 || { echo "  ERROR trellis2";       W1_FAIL=1; }
    echo "  Wave 1 done (fail=$W1_FAIL)"

    echo ""
    echo "===== Wave 2: Distilled H3D-2.1 (GPU 0-3 parallel) ====="
    ( cd "$PROJECT_DIR/models/hunyuan3d21/hy3dshape" && \
      PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 "$H3D_PY" \
        "$PROJECT_DIR/distill_methods/scripts/run_inference_distilled.py" \
        --config "$INFERENCE_CONFIG" --model-name pd_6step $MAX_SAMPLES_ARG \
    ) > "$LOG_DIR/inference_pd_6step.log" 2>&1 &
    P0=$!
    ( cd "$PROJECT_DIR/models/hunyuan3d21/hy3dshape" && \
      PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 "$H3D_PY" \
        "$PROJECT_DIR/distill_methods/scripts/run_inference_distilled.py" \
        --config "$INFERENCE_CONFIG" --model-name cd_4step $MAX_SAMPLES_ARG \
    ) > "$LOG_DIR/inference_cd_4step.log" 2>&1 &
    P1=$!
    ( cd "$PROJECT_DIR/models/hunyuan3d21/hy3dshape" && \
      PYTHONPATH=. CUDA_VISIBLE_DEVICES=2 "$H3D_PY" \
        "$PROJECT_DIR/distill_methods/scripts/run_inference_distilled.py" \
        --config "$INFERENCE_CONFIG" --model-name dmd1_1step $MAX_SAMPLES_ARG \
    ) > "$LOG_DIR/inference_dmd1_1step.log" 2>&1 &
    P2=$!
    ( cd "$PROJECT_DIR/models/hunyuan3d21/hy3dshape" && \
      PYTHONPATH=. CUDA_VISIBLE_DEVICES=3 "$H3D_PY" \
        "$PROJECT_DIR/distill_methods/scripts/run_inference_distilled.py" \
        --config "$INFERENCE_CONFIG" --model-name dmd2_1step $MAX_SAMPLES_ARG \
    ) > "$LOG_DIR/inference_dmd2_1step.log" 2>&1 &
    P3=$!
    echo "  GPU0 pd_6step   PID=$P0"
    echo "  GPU1 cd_4step   PID=$P1"
    echo "  GPU2 dmd1_1step PID=$P2"
    echo "  GPU3 dmd2_1step PID=$P3"
    W2_FAIL=0
    wait $P0 || { echo "  ERROR pd_6step";   W2_FAIL=1; }
    wait $P1 || { echo "  ERROR cd_4step";   W2_FAIL=1; }
    wait $P2 || { echo "  ERROR dmd1_1step"; W2_FAIL=1; }
    wait $P3 || { echo "  ERROR dmd2_1step"; W2_FAIL=1; }
    echo "  Wave 2 done (fail=$W2_FAIL)"

    echo ""
    echo "===== Wave 3: MDT-Dist (GPU 0) ====="
    ( cd "$PROJECT_DIR/models/trellis" && \
      SPCONV_ALGO=native PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 "$TRELLIS_PY" \
        "$PROJECT_DIR/pipeline/scripts/run_inference_mdt_dist.py" \
        --config "$INFERENCE_CONFIG" $MAX_SAMPLES_ARG \
    ) > "$LOG_DIR/inference_mdt_dist.log" 2>&1
    echo "  Wave 3 done"

    echo ""
    echo "===== Inference summary ====="
    for m in teacher_50step flashvdm pd_6step cd_4step dmd1_1step dmd2_1step trellis mdt_dist trellis2; do
        count=$(find "$RUN_DIR/predictions/$m" -name mesh_raw.obj 2>/dev/null | wc -l)
        echo "  $m: $count meshes"
    done
fi

# ---------- Phase 3: Evaluation ----------
if [[ "$SKIP_EVAL" == "false" ]]; then
    echo ""
    echo "===== Phase 3: Evaluation (Track A + B, default clamp [0.5, 2.0]) ====="
    "$EVAL_PY" "$PROJECT_DIR/pipeline/scripts/run_eval.py" \
        --config "$INFERENCE_CONFIG" \
        $MAX_SAMPLES_ARG \
        > "$LOG_DIR/eval_default.log" 2>&1 || echo "  eval had errors (see log)"

    if [[ "$SKIP_SENSITIVITY" == "false" ]]; then
        echo ""
        echo "===== Phase 4: Scale clamp sensitivity ([0.3, 3.0]) ====="
        SENS_CONFIG="$RUN_DIR/config_sensitivity.yaml"
        "$EVAL_PY" -c "
import yaml
with open('$INFERENCE_CONFIG') as f: cfg = yaml.safe_load(f)
# Metrics output to a subdir; predictions_root already absolute, reused
cfg['output_root'] = '$RUN_DIR/clamp_loose'
cfg['alignment']['track_a']['scale_clamp'] = [0.3, 3.0]
with open('$SENS_CONFIG', 'w') as f: yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
"
        "$EVAL_PY" "$PROJECT_DIR/pipeline/scripts/run_eval.py" \
            --config "$SENS_CONFIG" --skip-track-b \
            $MAX_SAMPLES_ARG \
            > "$LOG_DIR/eval_clamp_loose.log" 2>&1 || echo "  sensitivity eval had errors (see log)"
    fi
fi

echo ""
echo "=========================================="
echo "Benchmark done. RUN_DIR: $RUN_DIR"
echo "=========================================="
