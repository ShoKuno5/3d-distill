#!/usr/bin/env bash
# qzcli M1 production job: DMD2 training on HSSD 420 → Toys4k 105 eval
#
# Setup:
#   - 2 GPU H100 DDP (effective batch=8)
#   - 15000 steps with checkpointing every 1000
#   - Output: /inspire/.../sk5/scratch/distill_methods/m1_dmd2/
#
# Run via: qzcli create -n m1-dmd2 -c "bash <this-script>" \
#          -w ws-9dcc0e1f-... -g lcg-<H100 group> --instances 1 --spec <2x H100>

set -euo pipefail

# Mirror all stdout/stderr to a persistent gpfs log so it can be inspected
# from sk-train without going through qzcli watch.
SK5=/inspire/qb-ilm/project/qproject-assement/zhangkaipeng-24043/sk5
LOGDIR="$SK5/scratch/qzcli_logs"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/$(basename ${0%.*})_$(hostname)_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "[log mirror: $LOG]"

# Ensure runtime shared libs needed by pymeshlab / opencv-headless / blender
# are present (sk-train overlay has them; qzcli container image may not).
if ! ldconfig -p | grep -q "libGL.so.1"; then
    echo "[setup] installing libgl1 libsm6 libxrender1 libxfixes3 libxi6 libxkbcommon0 ..."
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq 2>&1 | tail -3 || true
    apt-get install -y -qq libgl1 libsm6 libxrender1 libxfixes3 libxi6 libxkbcommon0 2>&1 | tail -5
fi

SK5=/inspire/qb-ilm/project/qproject-assement/zhangkaipeng-24043/sk5
mkdir -p /root/.cache
[ ! -e /root/.cache/hy3dgen ]      && ln -sfn "$SK5/hy3dgen_cache" /root/.cache/hy3dgen
[ ! -e /root/.cache/huggingface ]  && ln -sfn "$SK5/hf_cache"      /root/.cache/huggingface
ls -la /root/.cache/ | head -5

REPO="$SK5/repos/3d-gen-eval"
H="$REPO/models/hunyuan3d21"
RUN_NAME="m1_dmd2_$(date +%Y%m%d_%H%M)"
OUTPUT_ROOT="$SK5/scratch/distill_methods/$RUN_NAME"

cd "$H/hy3dshape"
export HF_HOME="$SK5/hf_cache"
export PYTHONPATH=".:$REPO/distill_methods/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline  # qzcli container can not reach api.wandb.ai — log locally, sync from sk-train later

# Load WANDB key from sk5/.env (sourced via ~/.bashrc on sk-train; on qzcli node need explicit)
[ -f "$SK5/.env" ] && set -a && . "$SK5/.env" && set +a
: "${WANDB_API_KEY:?WANDB_API_KEY must be set (check sk5/.env or qzcli job env)}"

# Detect GPU count for DDP nproc_per_node
NGPU=$(nvidia-smi -L | wc -l)
echo "================================================================"
echo "M1 DMD2 training: $RUN_NAME"
echo "host: $(hostname)  ngpu: $NGPU"
echo "================================================================"

# Patch config: output_dir + ensure batch_size=4 (effective $NGPU*4 with DDP)
CFG="$REPO/distill_methods/configs/config_525_hssd.yaml"
RUN_CFG="$OUTPUT_ROOT/config.yaml"
mkdir -p "$OUTPUT_ROOT"
"$H/.venv/bin/python" -c "
import yaml
cfg = yaml.safe_load(open('$CFG'))
cfg['output_root'] = '$OUTPUT_ROOT'
cfg['training']['batch_size'] = 4
yaml.safe_dump(cfg, open('$RUN_CFG', 'w'), default_flow_style=False, sort_keys=False)
print('config: $RUN_CFG')
"

# Launch via torchrun (DDP if >1 GPU)
if [ "$NGPU" -gt 1 ]; then
    echo "Launching DDP with $NGPU GPUs"
    "$H/.venv/bin/torchrun" --nproc_per_node="$NGPU" \
        "$REPO/distill_methods/src/train.py" \
        --config "$RUN_CFG" \
        --method dmd2 \
        --output-dir "$OUTPUT_ROOT"
else
    echo "Single GPU run"
    "$H/.venv/bin/python" \
        "$REPO/distill_methods/src/train.py" \
        --config "$RUN_CFG" \
        --method dmd2 \
        --output-dir "$OUTPUT_ROOT"
fi

echo "================================================================"
echo "M1 DMD2 done: $(date -Iseconds)"
echo "checkpoints: $OUTPUT_ROOT/checkpoints/dmd2/"
echo "================================================================"
