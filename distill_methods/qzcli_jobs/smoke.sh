#!/usr/bin/env bash
# qzcli smoke job: verify env on H100 node before launching full training
#
# Checks:
#   1. gpfs visibility (/inspire/.../sk5/ readable)
#   2. CUDA + GPU count + memory
#   3. PyTorch + Hunyuan3D-2.1 pipeline importable
#   4. HSSD latent npz readable
#   5. Toys4k 105 input images present
#   6. train.py invocable with 5-step config (DMD2)
#
# Run via: qzcli create -n smoke-distill3d -c "bash <this-script>" \
#          -w ws-9dcc0e1f-... -g lcg-<H100 group> --instances 1

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

echo "================================================================"
echo "qzcli smoke job: $(date -Iseconds)"
echo "host: $(hostname), user: $(whoami)"
echo "================================================================"

# --- 1. gpfs visibility ---
echo
echo "[1/6] gpfs visibility"
ls -d "$SK5" >/dev/null && echo "  ✓ $SK5 readable"
ls -d "$REPO" >/dev/null && echo "  ✓ $REPO readable"
ls -d "$H/.venv" >/dev/null && echo "  ✓ existing .venv on gpfs"

# --- 2. CUDA / GPU ---
echo
echo "[2/6] CUDA / GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
NGPU=$(nvidia-smi -L | wc -l)
echo "  GPU count: $NGPU"

# --- 3. PyTorch + Hunyuan3D pipeline import ---
echo
echo "[3/6] PyTorch + Hunyuan3D-2.1 pipeline import"
cd "$H/hy3dshape"
export HF_HOME="$SK5/hf_cache"
export PYTHONPATH=".:$REPO/distill_methods/src"

"$H/.venv/bin/python" -c "
import torch
print(f'  torch:  {torch.__version__}  cuda={torch.cuda.is_available()}')
print(f'  device: {torch.cuda.get_device_name(0)}  mem={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
print('  ✓ Hunyuan3D-2.1 pipeline importable')
"

# --- 4. HSSD latent npz readable ---
echo
echo "[4/6] HSSD latent npz"
NPZ_DIR="$SK5/scratch/distill_methods/525_hssd/training_data"
NPZ_COUNT=$(ls "$NPZ_DIR" | wc -l)
echo "  npz count in $NPZ_DIR: $NPZ_COUNT (expected 420)"
"$H/.venv/bin/python" -c "
import numpy as np, os
d = sorted(os.listdir('$NPZ_DIR'))[0]
data = np.load(os.path.join('$NPZ_DIR', d))
print(f'  ✓ {d}: latent={data[\"latent\"].shape}, image_cond={data[\"image_cond\"].shape}')
"

# --- 5. Toys4k 105 eval input images ---
echo
echo "[5/6] Toys4k 105 eval renders"
IMG_COUNT=$(find "$SK5/data/toys4k/renders" -name image.png | wc -l)
echo "  image count: $IMG_COUNT (expected 105)"

# --- 6. train.py importable + 5-step smoke ---
echo
echo "[6/6] train.py smoke (5 steps DMD2)"
SMOKE_CFG="$REPO/distill_methods/configs/config_525_hssd_smoke.yaml"
"$H/.venv/bin/python" -c "
import yaml
cfg = yaml.safe_load(open('$REPO/distill_methods/configs/config_525_hssd.yaml'))
cfg['training']['total_steps'] = 5
cfg['training']['batch_size'] = 1
cfg['training']['log_interval'] = 1
cfg['training']['save_interval'] = 100
cfg['training']['methods']['dmd2']['replay_buffer_size'] = 32
cfg['training']['methods']['dmd2']['replay_warmup'] = 4
cfg['output_root'] = '$SK5/scratch/qzcli_smoke'
yaml.safe_dump(cfg, open('$SMOKE_CFG', 'w'), default_flow_style=False, sort_keys=False)
"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
CUDA_VISIBLE_DEVICES=0 "$H/.venv/bin/python" "$REPO/distill_methods/src/train.py" \
  --config "$SMOKE_CFG" --method dmd2 2>&1 | tail -15

echo
echo "================================================================"
echo "smoke job OK: $(date -Iseconds)"
echo "================================================================"
