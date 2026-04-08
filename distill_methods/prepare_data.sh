#!/usr/bin/env bash
# Data preparation for distillation experiment
#
# Generates all data needed before training:
#   1. Train/test manifests
#   2. Input image renders
#   3. VAE latent encoding (4 GPU parallel)
#   4. DMD1 regression pairs (4 GPU parallel)
#   5. Verification
#
# Usage:
#   bash distill_methods/prepare_data.sh
#   bash distill_methods/prepare_data.sh --max-samples 50

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="$SCRIPT_DIR/config.yaml"

# Parse args
MAX_SAMPLES_ARG=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --max-samples=*) MAX_SAMPLES_ARG="--max-samples ${1#*=}"; shift ;;
        --max-samples) MAX_SAMPLES_ARG="--max-samples $2"; shift 2 ;;
        *) echo "WARNING: Unknown argument: $1"; shift ;;
    esac
done

H3D_PYTHON="$PROJECT_DIR/envs/hunyuan3d-venv/bin/python"

OUTPUT_ROOT=$($H3D_PYTHON -c "import yaml; cfg=yaml.safe_load(open('$CONFIG')); print(cfg['output_root'])")
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR"

echo "================================================"
echo "Data Preparation for Distillation Experiment"
echo "================================================"
echo "Config: $CONFIG"
echo "Output: $OUTPUT_ROOT"
echo ""

# =========================================
# Step 1: Manifests
# =========================================
MANIFEST_TRAIN="$SCRIPT_DIR/manifest_train.csv"
MANIFEST_TEST="$SCRIPT_DIR/manifest_test.csv"
if [ ! -f "$MANIFEST_TRAIN" ] || [ ! -f "$MANIFEST_TEST" ]; then
    echo "--- Generating train/test manifests ---"
    $H3D_PYTHON "$SCRIPT_DIR/scripts/create_manifests.py" \
        --target-samples 525 --seed 42 \
        2>&1 | tee "$LOG_DIR/create_manifests.log"
else
    echo "Manifests already exist: $MANIFEST_TRAIN, $MANIFEST_TEST"
fi
echo ""

# =========================================
# Step 2: Input image renders
# =========================================
echo "--- Rendering input images (4 GPU) ---"
$H3D_PYTHON "$SCRIPT_DIR/scripts/render_batch.py" --num-gpus 4 \
    2>&1 | tee "$LOG_DIR/render_batch.log"
echo ""

# =========================================
# Step 3: VAE latent encoding (4 GPU parallel)
# =========================================
echo "--- Encoding training data (4 GPU shards) ---"
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
# Step 4: DMD1 regression pairs (4 GPU parallel)
# =========================================
echo "--- Generating DMD1 regression pairs (4 GPU shards) ---"
DMD1_PIDS=()
for SHARD in 0 1 2 3; do
    (
        cd "$PROJECT_DIR/models/hunyuan3d21/hy3dshape"
        PYTHONPATH=. CUDA_VISIBLE_DEVICES=$SHARD $H3D_PYTHON \
            $PROJECT_DIR/distill_methods/scripts/generate_dmd1_pairs.py \
            --config "$CONFIG" --shard $SHARD --num-shards 4 \
            2>&1 | tee "$LOG_DIR/generate_dmd1_pairs_shard${SHARD}.log"
    ) &
    DMD1_PIDS+=($!)
done
for PID in "${DMD1_PIDS[@]}"; do
    wait $PID || echo "WARNING: DMD1 pair generation shard failed (PID $PID)"
done
echo "DMD1 pair generation complete."
echo ""

# =========================================
# Step 5: Verification
# =========================================
echo "--- Data verification ---"
$H3D_PYTHON -c "
import csv, os, sys, glob, yaml, numpy as np

script_dir = '$SCRIPT_DIR'
cfg = yaml.safe_load(open(os.path.join(script_dir, 'config.yaml')))

# Check manifests
for name in ['manifest_train.csv', 'manifest_test.csv']:
    path = os.path.join(script_dir, name)
    with open(path) as f:
        rows = list(csv.DictReader(f))
    print(f'{name}: {len(rows)} samples')
    missing_img = [r['object_id'] for r in rows if not os.path.exists(r['input_image'])]
    missing_pc = [r['object_id'] for r in rows if not os.path.exists(r['point_cloud'])]
    missing_mesh = [r['object_id'] for r in rows if not os.path.exists(r['mesh_obj'])]
    if missing_img: print(f'  WARNING: {len(missing_img)} missing input images')
    if missing_pc:  print(f'  WARNING: {len(missing_pc)} missing point clouds')
    if missing_mesh: print(f'  WARNING: {len(missing_mesh)} missing mesh files')
    cats = {}
    for r in rows:
        cats[r['category']] = cats.get(r['category'], 0) + 1
    print(f'  Categories: {len(cats)}, min/max per cat: {min(cats.values())}/{max(cats.values())}')

# Check training data
td_dir = cfg['training']['training_data_dir']
npz_files = glob.glob(os.path.join(td_dir, '*.npz'))
train_path = os.path.join(script_dir, 'manifest_train.csv')
with open(train_path) as f:
    train_ids = {r['object_id'] for r in csv.DictReader(f)}
encoded_ids = {os.path.splitext(os.path.basename(f))[0] for f in npz_files}
missing_encoded = train_ids - encoded_ids
print(f'')
print(f'Training data: {len(npz_files)} NPZ files ({\"OK\" if not missing_encoded else f\"MISSING {len(missing_encoded)}\"})')

# Check DMD1 pairs
dmd1_cfg = cfg['training']['methods']['dmd1']
pairs_dir = dmd1_cfg['pairs_dir']
num_target = dmd1_cfg['num_pairs']
num_pairs = len([f for f in os.listdir(pairs_dir) if f.endswith('.npz')]) if os.path.isdir(pairs_dir) else 0
print(f'DMD1 pairs: {num_pairs}/{num_target} ({\"OK\" if num_pairs >= num_target else \"INCOMPLETE\"})')

# Summary
all_ok = not missing_encoded and num_pairs >= num_target
print(f'')
print(f'Status: {\"READY for training\" if all_ok else \"INCOMPLETE — fix issues above\"}')
" 2>&1 | tee "$LOG_DIR/data_verification.log"

echo ""
echo "Done. If status is READY, run training with:"
echo "  bash distill_methods/run.sh"
