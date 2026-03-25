#!/usr/bin/env bash
# Distill Profiling: Unified profiling + evaluation
#
# 4 models on 4 GPUs: trellis (v1 teacher), mdt_dist, hunyuan3d (v2.0 teacher), flashvdm
# N=48 samples. Block-level profiling + mesh saving + geometry/FD evaluation.
#
# Usage:
#   bash experiments/distill_profiling/run.sh
#   bash experiments/distill_profiling/run.sh --skip-inference
#   bash experiments/distill_profiling/run.sh --workers=4

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
CONFIG="$SCRIPT_DIR/config.yaml"

# Parse args
SKIP_INFERENCE=false
SKIP_RENDERS=false
WORKERS_ARG=""
MAX_SAMPLES_ARG=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-inference) SKIP_INFERENCE=true; shift ;;
        --skip-renders) SKIP_RENDERS=true; shift ;;
        --workers=*) WORKERS_ARG="--workers ${1#*=}"; shift ;;
        --workers) WORKERS_ARG="--workers $2"; shift 2 ;;
        --max-samples=*) MAX_SAMPLES_ARG="--max-samples ${1#*=}"; shift ;;
        --max-samples) MAX_SAMPLES_ARG="--max-samples $2"; shift 2 ;;
        *) echo "WARNING: Unknown argument: $1"; shift ;;
    esac
done

# Python interpreters
TRELLIS_PY="$PROJECT_DIR/envs/miniconda3/envs/trellis/bin/python"
H3D_PY="$PROJECT_DIR/envs/hunyuan3d-venv/bin/python"
EVAL_PY="$H3D_PY"

# Output
OUTPUT_ROOT=$($EVAL_PY -c "import yaml; cfg=yaml.safe_load(open('$CONFIG')); print(cfg['output_root'])")
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR"

echo "========================================="
echo "Distill Profiling: teacher vs distilled"
echo "========================================="
echo "Config: $CONFIG"
echo "Output: $OUTPUT_ROOT"
echo ""

# =========================================
# Phase 1: Profiled inference (4 models, 4 GPUs)
# =========================================
if [ "$SKIP_INFERENCE" = false ]; then
    echo "========================================="
    echo "Phase 1: Profiled inference (4 GPUs)"
    echo "========================================="

    # GPU 0: trellis v1 teacher (25×2 steps)
    (
        cd "$PROJECT_DIR/models/trellis"
        PYTHONPATH=. SPCONV_ALGO=native CUDA_VISIBLE_DEVICES=0 \
        $TRELLIS_PY \
        "$SCRIPT_DIR/run_profile_trellis.py" \
            --config "$CONFIG" $MAX_SAMPLES_ARG \
            2>&1 | tee "$LOG_DIR/profile_trellis.log"
    ) &
    PID_GPU0=$!

    # GPU 1: mdt_dist (2×2 steps)
    (
        cd "$PROJECT_DIR/models/trellis"
        PYTHONPATH=. SPCONV_ALGO=native CUDA_VISIBLE_DEVICES=1 \
        $TRELLIS_PY \
        "$SCRIPT_DIR/run_profile_mdt_dist.py" \
            --config "$CONFIG" $MAX_SAMPLES_ARG \
            2>&1 | tee "$LOG_DIR/profile_mdt_dist.log"
    ) &
    PID_GPU1=$!

    # GPU 2: hunyuan3d v2.0 teacher (30 steps)
    (
        cd "$PROJECT_DIR/models/hunyuan3d"
        PYTHONPATH=. \
        LD_PRELOAD=/opt/unreal-engine/usr/lib/x86_64-linux-gnu/libOpenGL.so.0 \
        CUDA_VISIBLE_DEVICES=2 \
        $H3D_PY \
        "$SCRIPT_DIR/run_profile_hunyuan3d.py" \
            --config "$CONFIG" $MAX_SAMPLES_ARG \
            2>&1 | tee "$LOG_DIR/profile_hunyuan3d.log"
    ) &
    PID_GPU2=$!

    # GPU 3: flashvdm (5 steps)
    (
        cd "$PROJECT_DIR/models/hunyuan3d"
        PYTHONPATH=. \
        LD_PRELOAD=/opt/unreal-engine/usr/lib/x86_64-linux-gnu/libOpenGL.so.0 \
        CUDA_VISIBLE_DEVICES=3 \
        $H3D_PY \
        "$SCRIPT_DIR/run_profile_flashvdm.py" \
            --config "$CONFIG" $MAX_SAMPLES_ARG \
            2>&1 | tee "$LOG_DIR/profile_flashvdm.log"
    ) &
    PID_GPU3=$!

    echo "  GPU0 (trellis):   PID $PID_GPU0  (25×2 steps)"
    echo "  GPU1 (mdt_dist):  PID $PID_GPU1  (2×2 steps)"
    echo "  GPU2 (hunyuan3d): PID $PID_GPU2  (30 steps)"
    echo "  GPU3 (flashvdm):  PID $PID_GPU3  (5 steps)"
    echo "  Logs: $LOG_DIR/"
    echo ""

    FAILED=0
    wait $PID_GPU0 || { echo "ERROR: trellis profiling failed"; FAILED=1; }
    wait $PID_GPU1 || { echo "ERROR: mdt_dist profiling failed"; FAILED=1; }
    wait $PID_GPU2 || { echo "ERROR: hunyuan3d profiling failed"; FAILED=1; }
    wait $PID_GPU3 || { echo "ERROR: flashvdm profiling failed"; FAILED=1; }

    if [ $FAILED -ne 0 ]; then
        echo "WARNING: Some profiling jobs failed. Continuing with available predictions..."
    fi
    echo "All profiling jobs finished."
    echo ""
fi

# =========================================
# Phase 2: Timing analysis
# =========================================
echo "========================================="
echo "Phase 2: Timing analysis"
echo "========================================="
cd "$PROJECT_DIR"
$EVAL_PY "$PROJECT_DIR/experiments/inference_profiling/analyze_timing.py" \
    --config "$CONFIG" \
    2>&1 | tee "$LOG_DIR/analyze_timing.log"
echo ""

# =========================================
# Phase 3: Multiview rendering
# =========================================
if [ "$SKIP_RENDERS" = false ]; then
    echo "========================================="
    echo "Phase 3: Multiview rendering"
    echo "========================================="
    cd "$PROJECT_DIR"

    MAX_SAMPLES_VAL=""
    if [ -n "$MAX_SAMPLES_ARG" ]; then
        MAX_SAMPLES_VAL="${MAX_SAMPLES_ARG##--max-samples }"
    fi

    MAX_SAMPLES_VAL="$MAX_SAMPLES_VAL" $EVAL_PY -c "
import yaml, sys, os
sys.path.insert(0, 'pipeline')
from src.evaluation.multiview_renderer import render_all_meshes
from src.utils.inference_config import load_and_filter_samples

cfg = yaml.safe_load(open('$CONFIG'))
max_s = os.environ.get('MAX_SAMPLES_VAL') or None
max_s = int(max_s) if max_s else None
samples = load_and_filter_samples(cfg, max_samples_override=max_s)
object_ids = [s.object_id for s in samples]
output_root = cfg['output_root']
renders_dir = output_root + '/multiview_renders'

# Render GT meshes
print('Rendering GT meshes...')
gt_count = 0
for s in samples:
    import os, trimesh
    from src.evaluation.multiview_renderer import render_multiview
    gt_dir = renders_dir + '/gt/' + s.object_id
    if all(os.path.exists(gt_dir + f'/view_{az}.png') for az in [0, 90, 180, 270]):
        gt_count += 1
        continue
    try:
        render_multiview(s.mesh_obj, gt_dir)
        gt_count += 1
    except Exception as e:
        print(f'  GT render failed for {s.object_id}: {e}')
print(f'  GT: {gt_count}/{len(samples)} rendered')

# Render predicted meshes for each model
for m in cfg['models']:
    model_name = m['name']
    print(f'Rendering {model_name} predictions...')
    count = render_all_meshes(
        m['predictions_root'],
        renders_dir + '/' + model_name,
        m['mesh_filename'],
        object_ids,
    )
    print(f'  {model_name}: {count}/{len(object_ids)} rendered')
" 2>&1 | tee "$LOG_DIR/multiview_render.log"
    echo ""
fi

# =========================================
# Phase 4: Geometry evaluation
# =========================================
echo "========================================="
echo "Phase 4: Geometry evaluation"
echo "========================================="
cd "$PROJECT_DIR"
$EVAL_PY pipeline/scripts/run_eval.py --config "$CONFIG" $MAX_SAMPLES_ARG $WORKERS_ARG \
    2>&1 | tee "$LOG_DIR/eval.log"
echo ""

# =========================================
# Phase 5: Frechet Distance evaluation
# =========================================
echo "========================================="
echo "Phase 5: Frechet Distance evaluation"
echo "========================================="
cd "$PROJECT_DIR"

MAX_SAMPLES_VAL=""
if [ -n "$MAX_SAMPLES_ARG" ]; then
    MAX_SAMPLES_VAL="${MAX_SAMPLES_ARG##--max-samples }"
fi

MAX_SAMPLES_VAL="$MAX_SAMPLES_VAL" $EVAL_PY -c "
import yaml, csv, os, sys
sys.path.insert(0, 'pipeline')
from src.evaluation.frechet_distance import compute_fd_for_model

cfg = yaml.safe_load(open('$CONFIG'))
output_root = cfg['output_root']
renders_dir = output_root + '/multiview_renders'
metrics_dir = output_root + '/metrics'
os.makedirs(metrics_dir, exist_ok=True)

fd_cfg = cfg.get('metrics', {}).get('frechet_distance', {})
if not fd_cfg.get('enabled', False):
    print('FD metrics disabled in config.')
    sys.exit(0)

fd_models = fd_cfg.get('models', ['inception_v3'])
gt_dir = renders_dir + '/gt'

rows = []
for model_cfg in cfg['models']:
    model_name = model_cfg['name']
    pred_dir = renders_dir + '/' + model_name
    if not os.path.isdir(pred_dir):
        print(f'  SKIP {model_name}: no renders found')
        continue

    for feat_model in fd_models:
        print(f'Computing FD for {model_name} ({feat_model})...')
        try:
            fd = compute_fd_for_model(pred_dir, gt_dir, model_name=feat_model)
            print(f'  {model_name} FD_{feat_model}: {fd:.4f}')
            rows.append({
                'model': model_name,
                'feature_extractor': feat_model,
                'frechet_distance': fd,
            })
        except Exception as e:
            print(f'  ERROR: {model_name}/{feat_model}: {e}')

if rows:
    fd_path = os.path.join(metrics_dir, 'frechet_distance.csv')
    with open(fd_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['model', 'feature_extractor', 'frechet_distance'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'FD results written to {fd_path}')
" 2>&1 | tee "$LOG_DIR/frechet_distance.log"
echo ""

# =========================================
# Phase 6: Report generation
# =========================================
echo "========================================="
echo "Phase 6: Report generation"
echo "========================================="
cd "$PROJECT_DIR"
$EVAL_PY pipeline/scripts/generate_report.py --config "$CONFIG" \
    2>&1 | tee "$LOG_DIR/report.log"

echo ""
echo "Done! Results at $OUTPUT_ROOT/"
