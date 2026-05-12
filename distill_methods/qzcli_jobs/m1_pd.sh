#!/usr/bin/env bash
# qzcli M1 PD (Progressive Distillation) training: HSSD 420 → Toys4k 105 eval
#
# Setup:
#   - 1 GPU H200 141GB (workspace lacks H100 spec)
#   - PD has 3 stages [50->25, 25->12, 12->6], each is its own train.py invocation
#     resuming from stage_{N-1}_merged/model.pt. Total ~3 × steps_per_stage steps.
#   - batch_size: 4 (same as DSW DMD2 baseline)
#
# Idempotent: stages with existing stage_{N}_merged/model.pt are skipped so this
# can be re-run to continue a partial PD run.
#
# Run via: qzcli create -n m1pd -c "bash <this-script>" \
#          -w ws-9dcc0e1f-... -g lcg-<group> --instances 1 --spec <1x GPU>

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

REPO="$SK5/repos/3d-gen-eval"
H="$REPO/models/hunyuan3d21"

# OUTPUT_ROOT: if M1_PD_OUTPUT_ROOT is exported (continuation case), reuse it.
# Otherwise create a new timestamped dir.
RUN_NAME="${M1_PD_RUN_NAME:-m1_pd_$(date +%Y%m%d_%H%M)}"
OUTPUT_ROOT="${M1_PD_OUTPUT_ROOT:-$SK5/scratch/distill_methods/$RUN_NAME}"

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
echo "M1 PD training: $RUN_NAME"
echo "output: $OUTPUT_ROOT"
echo "host: $(hostname)  ngpu: $NGPU  gpu: $GPU_NAME ($GPU_MEM)"
echo "================================================================"

CFG="$REPO/distill_methods/configs/config_525_hssd.yaml"
RUN_CFG="$OUTPUT_ROOT/config.yaml"
mkdir -p "$OUTPUT_ROOT"
if [ ! -f "$RUN_CFG" ]; then
    "$H/.venv/bin/python" -c "
import yaml
cfg = yaml.safe_load(open('$CFG'))
cfg['output_root'] = '$OUTPUT_ROOT'
cfg['training']['batch_size'] = 4
yaml.safe_dump(cfg, open('$RUN_CFG', 'w'), default_flow_style=False, sort_keys=False)
print('config: $RUN_CFG')
"
else
    echo "config exists: $RUN_CFG (reusing)"
fi

NSTAGES=$("$H/.venv/bin/python" -c "import yaml; print(len(yaml.safe_load(open('$RUN_CFG'))['training']['methods']['pd']['stages']))")
echo "PD: $NSTAGES stages total"
for STAGE in $(seq 0 $((NSTAGES - 1))); do
    MERGED="$OUTPUT_ROOT/checkpoints/pd/stage_${STAGE}_merged/model.pt"
    if [ -f "$MERGED" ]; then
        echo "[stage $STAGE] already complete (merged ckpt exists), skipping"
        continue
    fi
    echo "================================================================"
    echo "PD stage $STAGE start: $(date -Iseconds)"
    echo "================================================================"
    if [ "$NGPU" -gt 1 ]; then
        "$H/.venv/bin/torchrun" --nproc_per_node="$NGPU" \
            "$REPO/distill_methods/src/train.py" \
            --config "$RUN_CFG" \
            --method pd \
            --stage $STAGE \
            --output-dir "$OUTPUT_ROOT"
    else
        "$H/.venv/bin/python" \
            "$REPO/distill_methods/src/train.py" \
            --config "$RUN_CFG" \
            --method pd \
            --stage $STAGE \
            --output-dir "$OUTPUT_ROOT"
    fi
done

echo "================================================================"
echo "M1 PD done: $(date -Iseconds)"
echo "checkpoints: $OUTPUT_ROOT/checkpoints/pd/"
echo "================================================================"
