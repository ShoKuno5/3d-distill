#!/usr/bin/env bash
# Distill Methods Comparison: PD / CD / DMD1 / DMD2 on Hunyuan3D-2.1
#
# Full pipeline:
#   Phase 0: Dataset preparation (manifest + encode training data)
#   Phase 1: Training (PD stages, CD, DMD1 pairs + train, DMD2)
#   Phase 2: Inference (teacher + FlashVDM + 4 distilled models)
#   Phase 3: Multiview rendering + evaluation
#   Phase 4: Report generation
#
# Usage:
#   bash distill_methods/run.sh
#   bash distill_methods/run.sh --skip-training
#   bash distill_methods/run.sh --skip-inference --max-samples 5

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
SKIP_DATA_PREP=false
AUTO=false
WORKERS_ARG=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --max-samples=*) MAX_SAMPLES_ARG="--max-samples ${1#*=}"; shift ;;
        --max-samples) MAX_SAMPLES_ARG="--max-samples $2"; shift 2 ;;
        --skip-training) SKIP_TRAINING=true; shift ;;
        --skip-inference) SKIP_INFERENCE=true; shift ;;
        --skip-renders) SKIP_RENDERS=true; shift ;;
        --skip-data-prep) SKIP_DATA_PREP=true; shift ;;
        --auto) AUTO=true; shift ;;
        --workers=*) WORKERS_ARG="--workers ${1#*=}"; shift ;;
        --workers) WORKERS_ARG="--workers $2"; shift 2 ;;
        *) echo "WARNING: Unknown argument: $1"; shift ;;
    esac
done

# Python interpreters
H3D_PYTHON="$PROJECT_DIR/envs/hunyuan3d-venv/bin/python"
EVAL_PYTHON="$H3D_PYTHON"

# Extract output_root from config
OUTPUT_ROOT=$($EVAL_PYTHON -c "import yaml; cfg=yaml.safe_load(open('$CONFIG')); print(cfg['output_root'])")
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR"

echo "================================================"
echo "Distill Methods Comparison: PD / CD / DMD1 / DMD2"
echo "================================================"
echo "Config: $CONFIG"
echo "Output: $OUTPUT_ROOT"
echo ""

# =========================================
# Phase 0: Dataset preparation
# =========================================
if [ "$SKIP_DATA_PREP" = false ]; then
    echo "========================================="
    echo "Phase 0: Dataset preparation"
    echo "========================================="

    # Generate train/test manifests if they don't exist
    MANIFEST_TRAIN="$SCRIPT_DIR/manifest_train.csv"
    MANIFEST_TEST="$SCRIPT_DIR/manifest_test.csv"
    if [ ! -f "$MANIFEST_TRAIN" ] || [ ! -f "$MANIFEST_TEST" ]; then
        echo "Generating train/test manifests..."
        $EVAL_PYTHON "$SCRIPT_DIR/scripts/create_manifests.py" \
            --target-samples 525 --seed 42 \
            2>&1 | tee "$LOG_DIR/create_manifests.log"
    else
        echo "Manifests already exist: $MANIFEST_TRAIN, $MANIFEST_TEST"
    fi

    # Render input images for new samples
    echo "Rendering input images..."
    $EVAL_PYTHON "$SCRIPT_DIR/scripts/render_batch.py" --num-gpus 4 \
        2>&1 | tee "$LOG_DIR/render_batch.log"

    # Encode training data (VAE latents + image conditions) — 4 GPU parallel
    echo "Encoding training data (4 GPU shards)..."
    ENCODE_PIDS=()
    for SHARD in 0 1 2 3; do
        (
            cd "$PROJECT_DIR/models/hunyuan3d21/hy3dshape"
            PYTHONPATH=. CUDA_VISIBLE_DEVICES=$SHARD $H3D_PYTHON \
                $PROJECT_DIR/distill_methods/scripts/prepare_training_data.py \
                --config "$CONFIG" $MAX_SAMPLES_ARG \
                --shard $SHARD --num-shards 4 \
                2>&1 | tee "$LOG_DIR/prepare_training_data_shard${SHARD}.log"
        ) &
        ENCODE_PIDS+=($!)
    done
    for PID in "${ENCODE_PIDS[@]}"; do
        wait $PID || echo "WARNING: Encoding shard failed (PID $PID)"
    done
    echo "Training data encoding complete."
    echo ""

    # =========================================
    # Phase 0b: Data verification
    # =========================================
    echo "========================================="
    echo "Phase 0b: Data verification"
    echo "========================================="
    $EVAL_PYTHON -c "
import csv, os, sys, glob

script_dir = '$SCRIPT_DIR'
project_dir = '$PROJECT_DIR'

# Check manifests
for name in ['manifest_train.csv', 'manifest_test.csv']:
    path = os.path.join(script_dir, name)
    with open(path) as f:
        rows = list(csv.DictReader(f))
    print(f'{name}: {len(rows)} samples')

    # Verify input images exist
    missing_img = [r['object_id'] for r in rows if not os.path.exists(r['input_image'])]
    if missing_img:
        print(f'  WARNING: {len(missing_img)} missing input images: {missing_img[:5]}...')
    else:
        print(f'  All input images exist')

    # Verify point clouds exist
    missing_pc = [r['object_id'] for r in rows if not os.path.exists(r['point_cloud'])]
    if missing_pc:
        print(f'  WARNING: {len(missing_pc)} missing point clouds')
    else:
        print(f'  All point clouds exist')

    # Verify mesh files exist
    missing_mesh = [r['object_id'] for r in rows if not os.path.exists(r['mesh_obj'])]
    if missing_mesh:
        print(f'  WARNING: {len(missing_mesh)} missing mesh files')
    else:
        print(f'  All mesh files exist')

    # Category distribution
    cats = {}
    for r in rows:
        cats[r['category']] = cats.get(r['category'], 0) + 1
    print(f'  Categories: {len(cats)}, min/max per cat: {min(cats.values())}/{max(cats.values())}')

# Check training data
import yaml
cfg = yaml.safe_load(open(os.path.join(script_dir, 'config.yaml')))
td_dir = cfg['training']['training_data_dir']
npz_files = glob.glob(os.path.join(td_dir, '*.npz'))
print(f'')
print(f'Training data (encoded): {len(npz_files)} NPZ files in {td_dir}')

# Cross-check: all train manifest samples have encoded data
train_path = os.path.join(script_dir, 'manifest_train.csv')
with open(train_path) as f:
    train_ids = {r['object_id'] for r in csv.DictReader(f)}
encoded_ids = {os.path.splitext(os.path.basename(f))[0] for f in npz_files}
missing_encoded = train_ids - encoded_ids
if missing_encoded:
    print(f'  WARNING: {len(missing_encoded)} train samples not yet encoded: {sorted(missing_encoded)[:5]}...')
else:
    print(f'  All {len(train_ids)} train samples encoded')

# Quick sanity check on one NPZ file
import numpy as np
if npz_files:
    d = np.load(npz_files[0])
    print(f'  Sample NPZ keys: {list(d.keys())}')
    print(f'  latent shape: {d[\"latent\"].shape}, image_cond shape: {d[\"image_cond\"].shape}')

print('')
print('Data preparation summary:')
print(f'  Train: {len(train_ids)} samples')
print(f'  Test: {len(rows)} samples')  # last loaded was test
print(f'  Encoded: {len(npz_files)} NPZ files')
print(f'  Status: {\"READY\" if not missing_encoded else \"INCOMPLETE\"}')" 2>&1 | tee "$LOG_DIR/data_verification.log"

    echo ""

    if [ "$AUTO" = false ]; then
        echo "========================================="
        echo "Data preparation complete. Review the verification output above."
        echo "To continue with training, re-run with:"
        echo "  bash distill_methods/run.sh --skip-data-prep"
        echo "Or run the full pipeline without pause:"
        echo "  bash distill_methods/run.sh --auto"
        echo "========================================="
        exit 0
    fi
fi
echo ""

# =========================================
# Phase 1: Training
# =========================================
if [ "$SKIP_TRAINING" = false ]; then
    echo "========================================="
    echo "Phase 1: Distillation training"
    echo "========================================="

    TRAIN_CMD="$PROJECT_DIR/distill_methods/src/train.py"

    # --- PD: 3 stages (50->25->12->6) ---
    echo "--- Progressive Distillation (3 stages) ---"
    for STAGE in 0 1 2; do
        echo "  PD Stage $STAGE ..."
        (
            cd "$PROJECT_DIR/models/hunyuan3d21/hy3dshape"
            PYTHONPATH=.:$PROJECT_DIR/distill_methods/src \
            CUDA_VISIBLE_DEVICES=0 $H3D_PYTHON "$TRAIN_CMD" \
                --config "$CONFIG" --method pd --stage $STAGE \
                2>&1 | tee "$LOG_DIR/train_pd_stage${STAGE}.log"
        )
    done

    # --- CD ---
    echo "--- Consistency Distillation ---"
    (
        cd "$PROJECT_DIR/models/hunyuan3d21/hy3dshape"
        PYTHONPATH=.:$PROJECT_DIR/distill_methods/src \
        CUDA_VISIBLE_DEVICES=1 $H3D_PYTHON "$TRAIN_CMD" \
            --config "$CONFIG" --method cd \
            2>&1 | tee "$LOG_DIR/train_cd.log"
    ) &
    PID_CD=$!

    # --- DMD1: generate pairs first (4 GPU parallel), then train ---
    echo "--- DMD1: Generating regression pairs (4 GPU shards) ---"
    DMD1_PAIR_PIDS=()
    for SHARD in 0 1 2 3; do
        (
            cd "$PROJECT_DIR/models/hunyuan3d21/hy3dshape"
            PYTHONPATH=. CUDA_VISIBLE_DEVICES=$SHARD $H3D_PYTHON \
                $PROJECT_DIR/distill_methods/scripts/generate_dmd1_pairs.py \
                --config "$CONFIG" --shard $SHARD --num-shards 4 \
                2>&1 | tee "$LOG_DIR/generate_dmd1_pairs_shard${SHARD}.log"
        ) &
        DMD1_PAIR_PIDS+=($!)
    done
    for PID in "${DMD1_PAIR_PIDS[@]}"; do
        wait $PID || echo "WARNING: DMD1 pair generation shard failed (PID $PID)"
    done
    echo "DMD1 pair generation complete."

    echo "--- DMD1: Training ---"
    (
        cd "$PROJECT_DIR/models/hunyuan3d21/hy3dshape"
        PYTHONPATH=.:$PROJECT_DIR/distill_methods/src \
        CUDA_VISIBLE_DEVICES=2 $H3D_PYTHON "$TRAIN_CMD" \
            --config "$CONFIG" --method dmd1 \
            2>&1 | tee "$LOG_DIR/train_dmd1.log"
    ) &
    PID_DMD1=$!

    # --- DMD2 ---
    echo "--- DMD2: Training ---"
    (
        cd "$PROJECT_DIR/models/hunyuan3d21/hy3dshape"
        PYTHONPATH=.:$PROJECT_DIR/distill_methods/src \
        CUDA_VISIBLE_DEVICES=3 $H3D_PYTHON "$TRAIN_CMD" \
            --config "$CONFIG" --method dmd2 \
            2>&1 | tee "$LOG_DIR/train_dmd2.log"
    ) &
    PID_DMD2=$!

    # Wait for parallel jobs
    echo "  Waiting for CD (PID $PID_CD), DMD1 ($PID_DMD1), DMD2 ($PID_DMD2) ..."
    FAILED=0
    wait $PID_CD || { echo "ERROR: CD training failed"; FAILED=1; }
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

    # Teacher 50-step (GPU 0)
    (
        cd "$PROJECT_DIR/models/hunyuan3d21/hy3dshape"
        PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $H3D_PYTHON \
            $PROJECT_DIR/distill_methods/scripts/run_inference_distilled.py \
            --config "$CONFIG" --model-name teacher_50step $MAX_SAMPLES_ARG \
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
            --config "$CONFIG" $MAX_SAMPLES_ARG \
            2>&1 | tee "$LOG_DIR/inference_flashvdm.log"
    ) &
    PID_FV=$!

    # Distilled models (GPU 2-3, sequential per GPU)
    (
        for MODEL in pd_6step cd_4step; do
            cd "$PROJECT_DIR/models/hunyuan3d21/hy3dshape"
            PYTHONPATH=. CUDA_VISIBLE_DEVICES=2 $H3D_PYTHON \
                $PROJECT_DIR/distill_methods/scripts/run_inference_distilled.py \
                --config "$CONFIG" --model-name "$MODEL" $MAX_SAMPLES_ARG \
                2>&1 | tee "$LOG_DIR/inference_${MODEL}.log"
        done
    ) &
    PID_G2=$!

    (
        for MODEL in dmd1_1step dmd2_1step; do
            cd "$PROJECT_DIR/models/hunyuan3d21/hy3dshape"
            PYTHONPATH=. CUDA_VISIBLE_DEVICES=3 $H3D_PYTHON \
                $PROJECT_DIR/distill_methods/scripts/run_inference_distilled.py \
                --config "$CONFIG" --model-name "$MODEL" $MAX_SAMPLES_ARG \
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
cd "$PROJECT_DIR"
$EVAL_PYTHON pipeline/scripts/run_eval.py --config "$CONFIG" $MAX_SAMPLES_ARG $WORKERS_ARG \
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
from src.utils.inference_config import load_and_filter_samples

cfg = yaml.safe_load(open('$CONFIG'))
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
$EVAL_PYTHON pipeline/scripts/generate_report.py --config "$CONFIG" \
    2>&1 | tee "$LOG_DIR/report.log"

echo ""
echo "Done! Report: $OUTPUT_ROOT/report.md"
