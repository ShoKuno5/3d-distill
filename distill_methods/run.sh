#!/usr/bin/env bash
# Distill Methods Comparison: PD / CD / DMD1 / DMD2 on Hunyuan3D-2.1
#
# Training + evaluation pipeline. Assumes data is already prepared:
#   - Manifests:       distill_methods/manifest_{train,test}.csv
#   - VAE latents:     results/distill_methods/training_data/*.npz
#   - DMD1 pairs:      results/distill_methods/dmd1_pairs/*.npz
#
# Data preparation scripts (run individually before this):
#   scripts/create_manifests.py      — train/test split
#   scripts/render_batch.py          — input image renders
#   scripts/prepare_training_data.py — VAE latent encoding
#   scripts/generate_dmd1_pairs.py   — DMD1 regression pairs
#
# Each run gets its own timestamped directory under runs/.
#
# Usage:
#   bash distill_methods/run.sh
#   bash distill_methods/run.sh --skip-training
#   bash distill_methods/run.sh --skip-inference --max-samples 5
#   bash distill_methods/run.sh --run-id 20260406_0900  # resume a specific run

set -euo pipefail

# Wandb API key (override via environment if needed)
export WANDB_API_KEY="${WANDB_API_KEY:-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}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="$SCRIPT_DIR/config.yaml"

# Parse args
MAX_SAMPLES_ARG=""
SKIP_TRAINING=false
SKIP_INFERENCE=false
SKIP_RENDERS=false
WORKERS_ARG=""
RUN_ID=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --max-samples=*) MAX_SAMPLES_ARG="--max-samples ${1#*=}"; shift ;;
        --max-samples) MAX_SAMPLES_ARG="--max-samples $2"; shift 2 ;;
        --skip-training) SKIP_TRAINING=true; shift ;;
        --skip-inference) SKIP_INFERENCE=true; shift ;;
        --skip-renders) SKIP_RENDERS=true; shift ;;
        --workers=*) WORKERS_ARG="--workers ${1#*=}"; shift ;;
        --workers) WORKERS_ARG="--workers $2"; shift 2 ;;
        --run-id=*) RUN_ID="${1#*=}"; shift ;;
        --run-id) RUN_ID="$2"; shift 2 ;;
        *) echo "WARNING: Unknown argument: $1"; shift ;;
    esac
done

# Python interpreters
H3D_PYTHON="$PROJECT_DIR/envs/hunyuan3d-venv/bin/python"
EVAL_PYTHON="$H3D_PYTHON"

# Extract base output_root from config
OUTPUT_BASE=$($EVAL_PYTHON -c "import yaml; cfg=yaml.safe_load(open('$CONFIG')); print(cfg['output_root'])")

# Run directory: each run gets a unique timestamped directory
if [ -z "$RUN_ID" ]; then
    RUN_ID=$(date +%Y%m%d_%H%M)
fi
RUN_DIR="$OUTPUT_BASE/runs/$RUN_ID"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"

# Record git branch and commit for traceability
BRANCH=$(git -C "$PROJECT_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
COMMIT=$(git -C "$PROJECT_DIR" rev-parse --short HEAD 2>/dev/null || echo "unknown")
cat > "$RUN_DIR/run_info.txt" <<EOF
run_id: $RUN_ID
branch: $BRANCH
commit: $COMMIT
config: $CONFIG
started: $(date -Iseconds)
command: $0 $@
EOF

echo "================================================"
echo "Distill Methods Comparison: PD / CD / DMD1 / DMD2"
echo "================================================"
echo "Config:  $CONFIG"
echo "Run ID:  $RUN_ID"
echo "Run dir: $RUN_DIR"
echo "Branch:  $BRANCH ($COMMIT)"
echo ""

# =========================================
# Phase 1: Training
# =========================================
if [ "$SKIP_TRAINING" = false ]; then
    echo "========================================="
    echo "Phase 1: Distillation training"
    echo "========================================="

    TRAIN_CMD="$PROJECT_DIR/distill_methods/src/train.py"

    # All 4 methods run in parallel: PD(GPU0), CD(GPU1), DMD1(GPU2), DMD2(GPU3)

    # --- PD: 3 stages sequential on GPU 0 ---
    echo "--- Progressive Distillation (3 stages, GPU 0) ---"
    (
        cd "$PROJECT_DIR/models/hunyuan3d21/hy3dshape"
        for STAGE in 0 1 2; do
            echo "=== PD Stage $STAGE ===" | tee -a "$LOG_DIR/train_pd_all.log"
            PYTHONPATH=.:$PROJECT_DIR/distill_methods/src \
            CUDA_VISIBLE_DEVICES=0 $H3D_PYTHON "$TRAIN_CMD" \
                --config "$CONFIG" --method pd --stage $STAGE \
                --output-dir "$RUN_DIR" \
                2>&1 | tee "$LOG_DIR/train_pd_stage${STAGE}.log" \
                     | tee -a "$LOG_DIR/train_pd_all.log"
        done
    ) &
    PID_PD=$!

    # --- CD on GPU 1 ---
    echo "--- Consistency Distillation (GPU 1) ---"
    (
        cd "$PROJECT_DIR/models/hunyuan3d21/hy3dshape"
        PYTHONPATH=.:$PROJECT_DIR/distill_methods/src \
        CUDA_VISIBLE_DEVICES=1 $H3D_PYTHON "$TRAIN_CMD" \
            --config "$CONFIG" --method cd \
            --output-dir "$RUN_DIR" \
            2>&1 | tee "$LOG_DIR/train_cd.log"
    ) &
    PID_CD=$!

    # --- DMD1 on GPU 2 ---
    echo "--- DMD1: Training (GPU 2) ---"
    (
        cd "$PROJECT_DIR/models/hunyuan3d21/hy3dshape"
        PYTHONPATH=.:$PROJECT_DIR/distill_methods/src \
        CUDA_VISIBLE_DEVICES=2 $H3D_PYTHON "$TRAIN_CMD" \
            --config "$CONFIG" --method dmd1 \
            --output-dir "$RUN_DIR" \
            2>&1 | tee "$LOG_DIR/train_dmd1.log"
    ) &
    PID_DMD1=$!

    # --- DMD2 on GPU 3 ---
    echo "--- DMD2: Training (GPU 3) ---"
    (
        cd "$PROJECT_DIR/models/hunyuan3d21/hy3dshape"
        PYTHONPATH=.:$PROJECT_DIR/distill_methods/src \
        CUDA_VISIBLE_DEVICES=3 $H3D_PYTHON "$TRAIN_CMD" \
            --config "$CONFIG" --method dmd2 \
            --output-dir "$RUN_DIR" \
            2>&1 | tee "$LOG_DIR/train_dmd2.log"
    ) &
    PID_DMD2=$!

    echo ""
    echo "  GPU 0: PD (3 stages sequential)  [PID $PID_PD]"
    echo "  GPU 1: CD                         [PID $PID_CD]"
    echo "  GPU 2: DMD1                       [PID $PID_DMD1]"
    echo "  GPU 3: DMD2                       [PID $PID_DMD2]"
    echo ""

    # Wait for all parallel jobs
    FAILED=0
    wait $PID_PD   || { echo "ERROR: PD training failed"; FAILED=1; }
    wait $PID_CD   || { echo "ERROR: CD training failed"; FAILED=1; }
    wait $PID_DMD1 || { echo "ERROR: DMD1 training failed"; FAILED=1; }
    wait $PID_DMD2 || { echo "ERROR: DMD2 training failed"; FAILED=1; }

    if [ $FAILED -ne 0 ]; then
        echo "WARNING: Some training jobs failed. Continuing with available checkpoints..."
    fi
    echo "All training jobs finished."
    echo ""
fi

# =========================================
# Phase 2: Inference
# =========================================
if [ "$SKIP_INFERENCE" = false ]; then
    echo "========================================="
    echo "Phase 2: Inference (6 models)"
    echo "========================================="

    # Generate run-specific config with output_root pointing to this run
    # (predictions_root and lora_path are resolved automatically by resolve_model_paths)
    INFERENCE_CONFIG="$RUN_DIR/config_inference.yaml"
    $EVAL_PYTHON -c "
import yaml
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
cfg['output_root'] = '$RUN_DIR'
with open('$INFERENCE_CONFIG', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
print('Inference config: $INFERENCE_CONFIG')
"

    # Teacher 50-step (GPU 0)
    (
        cd "$PROJECT_DIR/models/hunyuan3d21/hy3dshape"
        PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $H3D_PYTHON \
            $PROJECT_DIR/distill_methods/scripts/run_inference_distilled.py \
            --config "$INFERENCE_CONFIG" --model-name teacher_50step $MAX_SAMPLES_ARG \
            2>&1 | tee "$LOG_DIR/inference_teacher.log"
    ) &
    PID_T=$!

    # FlashVDM (GPU 1)
    (
        cd "$PROJECT_DIR/models/hunyuan3d21/hy3dshape"
        PYTHONPATH=. \
        LD_PRELOAD=/opt/unreal-engine/usr/lib/x86_64-linux-gnu/libOpenGL.so.0 \
        CUDA_VISIBLE_DEVICES=1 $H3D_PYTHON \
            $PROJECT_DIR/pipeline/scripts/run_inference_flashvdm.py \
            --config "$INFERENCE_CONFIG" $MAX_SAMPLES_ARG \
            2>&1 | tee "$LOG_DIR/inference_flashvdm.log"
    ) &
    PID_FV=$!

    # Distilled models (GPU 2-3, sequential per GPU)
    (
        for MODEL in pd_6step cd_4step; do
            cd "$PROJECT_DIR/models/hunyuan3d21/hy3dshape"
            PYTHONPATH=. CUDA_VISIBLE_DEVICES=2 $H3D_PYTHON \
                $PROJECT_DIR/distill_methods/scripts/run_inference_distilled.py \
                --config "$INFERENCE_CONFIG" --model-name "$MODEL" $MAX_SAMPLES_ARG \
                2>&1 | tee "$LOG_DIR/inference_${MODEL}.log"
        done
    ) &
    PID_G2=$!

    (
        for MODEL in dmd1_1step dmd2_1step; do
            cd "$PROJECT_DIR/models/hunyuan3d21/hy3dshape"
            PYTHONPATH=. CUDA_VISIBLE_DEVICES=3 $H3D_PYTHON \
                $PROJECT_DIR/distill_methods/scripts/run_inference_distilled.py \
                --config "$INFERENCE_CONFIG" --model-name "$MODEL" $MAX_SAMPLES_ARG \
                2>&1 | tee "$LOG_DIR/inference_${MODEL}.log"
        done
    ) &
    PID_G3=$!

    echo "  GPU0: teacher_50step  (PID $PID_T)"
    echo "  GPU1: flashvdm        (PID $PID_FV)"
    echo "  GPU2: pd_6step, cd_4step (PID $PID_G2)"
    echo "  GPU3: dmd1_1step, dmd2_1step (PID $PID_G3)"

    FAILED=0
    wait $PID_T  || { echo "ERROR: teacher inference failed"; FAILED=1; }
    wait $PID_FV || { echo "ERROR: flashvdm inference failed"; FAILED=1; }
    wait $PID_G2 || { echo "ERROR: GPU2 inference failed"; FAILED=1; }
    wait $PID_G3 || { echo "ERROR: GPU3 inference failed"; FAILED=1; }

    if [ $FAILED -ne 0 ]; then
        echo "WARNING: Some inference jobs failed. Continuing with available predictions..."
    fi
    echo "All inference jobs finished."
    echo ""
fi

# =========================================
# Phase 3: Evaluation
# =========================================
echo "========================================="
echo "Phase 3: Geometry evaluation"
echo "========================================="
EVAL_CONFIG="${INFERENCE_CONFIG:-$CONFIG}"
cd "$PROJECT_DIR"
$EVAL_PYTHON pipeline/scripts/run_eval.py --config "$EVAL_CONFIG" $MAX_SAMPLES_ARG $WORKERS_ARG \
    2>&1 | tee "$LOG_DIR/eval.log"

echo ""

# Multiview rendering + Frechet Distance
if [ "$SKIP_RENDERS" = false ]; then
    echo "========================================="
    echo "Phase 3b: Multiview rendering + FD"
    echo "========================================="

    MAX_SAMPLES_VAL=""
    if [ -n "$MAX_SAMPLES_ARG" ]; then
        MAX_SAMPLES_VAL="${MAX_SAMPLES_ARG##--max-samples }"
    fi

    MAX_SAMPLES_VAL="$MAX_SAMPLES_VAL" $EVAL_PYTHON -c "
import yaml, csv, os, sys
sys.path.insert(0, 'pipeline')
from src.evaluation.multiview_renderer import render_all_meshes
from src.evaluation.frechet_distance import compute_fd_for_model
from src.utils.inference_config import load_and_filter_samples, resolve_model_paths

cfg = yaml.safe_load(open('$EVAL_CONFIG'))
resolve_model_paths(cfg)
max_s = os.environ.get('MAX_SAMPLES_VAL') or None
max_s = int(max_s) if max_s else None
samples = load_and_filter_samples(cfg, max_samples_override=max_s, manifest_key='test_manifest')
object_ids = [s.object_id for s in samples]
output_root = cfg['output_root']
renders_dir = output_root + '/multiview_renders'

# Render GT + predictions
from src.evaluation.multiview_renderer import render_multiview
for s in samples:
    gt_dir = renders_dir + '/gt/' + s.object_id
    if not all(os.path.exists(gt_dir + f'/view_{az}.png') for az in [0,90,180,270]):
        try: render_multiview(s.mesh_obj, gt_dir)
        except Exception as e: print(f'  GT failed: {s.object_id}: {e}')
for m in cfg['models']:
    render_all_meshes(m['predictions_root'], renders_dir+'/'+m['name'], m['mesh_filename'], object_ids)

# FD
fd_cfg = cfg.get('metrics',{}).get('frechet_distance',{})
if fd_cfg.get('enabled'):
    metrics_dir = output_root + '/metrics'
    os.makedirs(metrics_dir, exist_ok=True)
    rows = []
    for mcfg in cfg['models']:
        pred_dir = renders_dir + '/' + mcfg['name']
        if not os.path.isdir(pred_dir): continue
        for feat in fd_cfg.get('models', ['inception_v3']):
            try:
                fd = compute_fd_for_model(pred_dir, renders_dir+'/gt', model_name=feat)
                print(f'{mcfg[\"name\"]} FD_{feat}: {fd:.4f}')
                rows.append({'model': mcfg['name'], 'feature_extractor': feat, 'frechet_distance': fd})
            except Exception as e:
                print(f'  FD error {mcfg[\"name\"]}/{feat}: {e}')
    if rows:
        fd_path = os.path.join(metrics_dir, 'frechet_distance.csv')
        with open(fd_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['model','feature_extractor','frechet_distance'])
            w.writeheader(); w.writerows(rows)
        print(f'FD results: {fd_path}')
" 2>&1 | tee "$LOG_DIR/multiview_fd.log"
fi

echo ""

# =========================================
# Phase 4: Report
# =========================================
echo "========================================="
echo "Phase 4: Report generation"
echo "========================================="
cd "$PROJECT_DIR"
$EVAL_PYTHON pipeline/scripts/generate_report.py --config "$EVAL_CONFIG" \
    2>&1 | tee "$LOG_DIR/report.log"

# Record completion
echo "completed: $(date -Iseconds)" >> "$RUN_DIR/run_info.txt"

echo ""
echo "Done!"
echo "  Run dir: $RUN_DIR"
echo "  Report:  $RUN_DIR/report.md"
