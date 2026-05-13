#!/usr/bin/env bash
# qzcli M1 DMD1 (Distribution Matching Distillation, multistep) training:
#   HSSD 420 → Toys4k 105 eval
#
# Setup:
#   - 1 GPU H200 141GB (DMD1 has fake_score similar to DMD2, needs H200)
#   - Pair generation FIRST (~1-2h, 20K teacher pairs from training data)
#   - Then 15000 steps distillation training (~25h)
#   - batch_size: 4 (same as DSW DMD2 baseline)
#
# Run via: qzcli create -n m1dmd1 -c "bash <this-script>" \
#          -w ws-9dcc0e1f-... -g lcg-<H200 group> --instances 1 --spec <1x H200>

set -euo pipefail

SK5=/inspire/qb-ilm/project/qproject-assement/zhangkaipeng-24043/sk5
LOGDIR="$SK5/scratch/qzcli_logs"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/$(basename ${0%.*})_$(hostname)_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "[log mirror: $LOG]"

if ! ldconfig -p | grep -q "libGL.so.1"; then
    echo "[setup] installing libgl1 libsm6 libxrender1 libxfixes3 libxi6 libxkbcommon0 ..."
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq 2>&1 | tail -3 || true
    apt-get install -y -qq libgl1 libsm6 libxrender1 libxfixes3 libxi6 libxkbcommon0 2>&1 | tail -5
fi

mkdir -p /root/.cache
[ ! -e /root/.cache/hy3dgen ]     && ln -sfn "$SK5/hy3dgen_cache" /root/.cache/hy3dgen
[ ! -e /root/.cache/huggingface ] && ln -sfn "$SK5/hf_cache"      /root/.cache/huggingface

REPO="$SK5/repos/3d-distill"
H="$REPO/models/hunyuan3d21"
RUN_NAME="m1_dmd1_$(date +%Y%m%d_%H%M)"
OUTPUT_ROOT="$SK5/scratch/distill_methods/$RUN_NAME"

cd "$H/hy3dshape"
export HF_HOME="$SK5/hf_cache"
export PYTHONPATH=".:$REPO/distill_methods/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

[ -f "$SK5/.env" ] && set -a && . "$SK5/.env" && set +a
: "${WANDB_API_KEY:?WANDB_API_KEY must be set}"

NGPU=$(nvidia-smi -L | wc -l)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
echo "================================================================"
echo "M1 DMD1 training: $RUN_NAME"
echo "host: $(hostname)  ngpu: $NGPU  gpu: $GPU_NAME ($GPU_MEM)"
echo "================================================================"

# Use SHARED pairs_dir at scale level (525_hssd/dmd1_pairs/) so subsequent
# DMD1 runs on the same scale can reuse the pairs.
PAIRS_DIR="$SK5/scratch/distill_methods/525_hssd/dmd1_pairs"

CFG="$REPO/distill_methods/configs/config_525_hssd.yaml"
RUN_CFG="$OUTPUT_ROOT/config.yaml"
mkdir -p "$OUTPUT_ROOT" "$PAIRS_DIR"
"$H/.venv/bin/python" -c "
import yaml
cfg = yaml.safe_load(open('$CFG'))
cfg['output_root'] = '$OUTPUT_ROOT'
cfg['training']['batch_size'] = 4
cfg['training']['methods']['dmd1']['pairs_dir'] = '$PAIRS_DIR'
yaml.safe_dump(cfg, open('$RUN_CFG', 'w'), default_flow_style=False, sort_keys=False)
print('config: $RUN_CFG')
"

# Step 1: pair generation (skip if 20K pairs already exist)
NUM_PAIRS=$("$H/.venv/bin/python" -c "import yaml; c=yaml.safe_load(open('$RUN_CFG')); print(c['training']['methods']['dmd1']['num_pairs'])")
HAVE_PAIRS=$(ls "$PAIRS_DIR"/pair_*.npz 2>/dev/null | wc -l)
echo "DMD1 pairs: have=$HAVE_PAIRS  need=$NUM_PAIRS"
if [ "$HAVE_PAIRS" -lt "$NUM_PAIRS" ]; then
    echo "Generating DMD1 pairs (single shard, may take 1-2h on H200)..."
    "$H/.venv/bin/python" \
        "$REPO/distill_methods/scripts/generate_dmd1_pairs.py" \
        --config "$RUN_CFG" --shard 0 --num-shards 1
    HAVE_PAIRS=$(ls "$PAIRS_DIR"/pair_*.npz 2>/dev/null | wc -l)
    echo "After pair gen: have=$HAVE_PAIRS"
fi

# Step 2: training
echo "================================================================"
echo "DMD1 training start: $(date -Iseconds)"
echo "================================================================"
if [ "$NGPU" -gt 1 ]; then
    "$H/.venv/bin/torchrun" --nproc_per_node="$NGPU" \
        "$REPO/distill_methods/src/train.py" \
        --config "$RUN_CFG" \
        --method dmd1 \
        --output-dir "$OUTPUT_ROOT"
else
    "$H/.venv/bin/python" \
        "$REPO/distill_methods/src/train.py" \
        --config "$RUN_CFG" \
        --method dmd1 \
        --output-dir "$OUTPUT_ROOT"
fi

echo "================================================================"
echo "M1 DMD1 done: $(date -Iseconds)"
echo "checkpoints: $OUTPUT_ROOT/checkpoints/dmd1/"
echo "================================================================"
