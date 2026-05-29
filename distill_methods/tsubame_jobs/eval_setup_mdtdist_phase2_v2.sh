#!/usr/bin/env bash
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=3:00:00
#$ -N mdtdist-phase2-v2
#$ -j y
#$ -o /gs/fs/tga-koike-shanda2/sk/scratch/tsubame_logs/eval_mdtdist_phase2_v2.qsub.log
#
# TRELLIS v1 (mdt_dist) Phase 2 v2:
#   - Lock torch==2.8.0+cu128 (xformers was upgrading to 2.12+cu130, breaking flash_attn)
#   - Re-init recursive submodules
#   - Install xformers compatible with torch 2.8
#   - Force reinstall flash_attn prebuilt after any deps mess

source "${COMMON_SH:-/gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/tsubame_jobs/_common.sh}"
tsubame_log_init "eval_mdtdist_phase2_v2"
tsubame_setup_env

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/gs/fs/tga-koike-shanda2/sk/cache/uv
export TORCH_HOME=/gs/fs/tga-koike-shanda2/sk/cache/torch

VENV_PY=/gs/fs/tga-koike-shanda2/sk/envs/mdtdist_uv/bin/python
REPO_DIR=/gs/fs/tga-koike-shanda2/sk/models/trellis_v1
PIP_CMD="uv pip install --python $VENV_PY --no-build-isolation"
PIP_CMD_REG="uv pip install --python $VENV_PY"

echo "================================================================"
echo "mdtdist Phase 2 v2: $(date -Iseconds), host=$(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader -i 0
nvcc --version | head -4
echo "================================================================"
set +e

echo ""
echo "[1/5] Reinstate torch 2.8 + cu128 (was upgraded to 2.12 by xformers)"
$PIP_CMD_REG --reinstall torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -5

echo ""
echo "[2/5] Reinstall flash-attn prebuilt (broken by torch upgrade)"
$PIP_CMD_REG --reinstall "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.7.4%2Bcu128torch2.8-cp310-cp310-linux_x86_64.whl" 2>&1 | tail -5

echo ""
echo "[3/5] Re-init recursive submodules"
cd $REPO_DIR
git submodule update --init --recursive 2>&1 | tail -10

echo ""
echo "[4/5] Try submodule extension builds"
for ext in diffoctreerast mip-splatting vox2seq; do
    EXTD=$(find $REPO_DIR -maxdepth 5 -name "$ext" -type d 2>/dev/null | head -1)
    if [ -z "$EXTD" ]; then echo "  $ext: not found"; continue; fi
    if [ -f "$EXTD/setup.py" ]; then
        echo "  building $ext"
        $PIP_CMD -e "$EXTD" 2>&1 | tail -5
    fi
done

# nvdiffrast separately (no submodule, regular install)
echo ""
echo "[4b] nvdiffrast"
$PIP_CMD_REG "git+https://github.com/NVlabs/nvdiffrast.git" 2>&1 | tail -5

echo ""
echo "[5/5] Sanity"
$VENV_PY -c "
import torch
print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())
for mod in ['xformers', 'nvdiffrast', 'spconv', 'flash_attn', 'trimesh']:
    try:
        m = __import__(mod); v = getattr(m, '__version__', '?')
        print(f'  {mod}: {v}')
    except Exception as e:
        print(f'  {mod}: FAIL {e}')
"
echo "================================================================"
echo "mdtdist v2 done: $(date -Iseconds)"
echo "================================================================"
