#!/usr/bin/env bash
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=1:00:00
#$ -N tsubame-smoke
#$ -j y
#$ -o /gs/fs/tga-koike-shanda2/sk/scratch/tsubame_logs/smoke.qsub.log
#
# TSUBAME smoke job: verify env on H100 96GB node before launching M1 production.
#
# Checks:
#   1. /gs/ visibility (SSD + HDD) and repo presence
#   2. CUDA + GPU count + memory
#   3. PyTorch + Hunyuan3D-2.1 pipeline importable
#   4. HSSD latent npz readable
#   5. Toys4k 105 eval renders present
#   6. train.py invocable with 5-step DMD2 smoke
#
# Run via interactive (after `qrsh -l node_q=1 -ar 6925 -g tga-koike-shanda`):
#   bash <repo>/distill_methods/tsubame_jobs/smoke.sh
# or batch:
#   qsub -ar 6925 -g tga-koike-shanda smoke.sh

source "$(dirname "$0")/_common.sh"
tsubame_log_init "smoke"
tsubame_setup_env

echo "================================================================"
echo "TSUBAME smoke: $(date -Iseconds)"
echo "host: $(hostname), user: $(whoami)"
echo "================================================================"

# --- 1. filesystem visibility ---
echo
echo "[1/6] filesystem visibility"
ls -d "$TSUBAME_ROOT" >/dev/null && echo "  OK $TSUBAME_ROOT readable"
ls -d "$REPO" >/dev/null && echo "  OK $REPO readable"
ls -d "$VENV" >/dev/null && echo "  OK $VENV present"

# --- 2. CUDA / GPU ---
echo
echo "[2/6] CUDA / GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
NGPU=$(nvidia-smi -L | wc -l)
echo "  GPU count: $NGPU"

# --- 3. PyTorch + Hunyuan3D pipeline import ---
echo
echo "[3/6] PyTorch + Hunyuan3D-2.1 pipeline import"
cd "$REPO/models/hunyuan3d21/hy3dshape"
"$PY" -c "
import torch
print(f'  torch:  {torch.__version__}  cuda={torch.cuda.is_available()}')
print(f'  device: {torch.cuda.get_device_name(0)}  mem={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
print('  OK Hunyuan3D-2.1 pipeline importable')
"

# --- 4. HSSD latent npz readable ---
echo
echo "[4/6] HSSD latent npz"
NPZ_DIR="$TSUBAME_ROOT/data/525_hssd/training_data"
NPZ_COUNT=$(ls "$NPZ_DIR" 2>/dev/null | wc -l)
echo "  npz count in $NPZ_DIR: $NPZ_COUNT (expected ~420)"
"$PY" -c "
import numpy as np, os
d = sorted(os.listdir('$NPZ_DIR'))[0]
data = np.load(os.path.join('$NPZ_DIR', d))
print(f'  OK {d}: latent={data[\"latent\"].shape}, image_cond={data[\"image_cond\"].shape}')
"

# --- 5. Toys4k 105 eval renders ---
echo
echo "[5/6] Toys4k 105 eval renders"
IMG_COUNT=$(find "$TSUBAME_ROOT/data/toys4k/renders" -name image.png 2>/dev/null | wc -l)
echo "  image count: $IMG_COUNT (expected 105)"

# --- 6. train.py 5-step DMD2 smoke ---
echo
echo "[6/6] train.py smoke (5-step DMD2)"
SMOKE_CFG="$REPO/distill_methods/configs/config_525_hssd_smoke_tsubame.yaml"
"$PY" - <<PY
import yaml
src = '$REPO/distill_methods/configs/config_525_hssd_tsubame.yaml'
cfg = yaml.safe_load(open(src))
cfg['training']['total_steps'] = 5
cfg['training']['batch_size'] = 1
cfg['training']['log_interval'] = 1
cfg['training']['save_interval'] = 100
cfg['training']['methods']['dmd2']['replay_buffer_size'] = 32
cfg['training']['methods']['dmd2']['replay_warmup'] = 4
cfg['output_root'] = '$TSUBAME_ROOT/scratch/tsubame_smoke'
yaml.safe_dump(cfg, open('$SMOKE_CFG', 'w'), default_flow_style=False, sort_keys=False)
PY

CUDA_VISIBLE_DEVICES=0 "$PY" "$REPO/distill_methods/src/train.py" \
  --config "$SMOKE_CFG" --method dmd2 2>&1 | tail -20

echo
echo "================================================================"
echo "smoke OK: $(date -Iseconds)"
echo "================================================================"
