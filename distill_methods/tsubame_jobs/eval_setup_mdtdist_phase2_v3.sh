#!/usr/bin/env bash
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=2:00:00
#$ -N mdtdist-phase2-v3
#$ -j y
#$ -o /gs/fs/tga-koike-shanda2/sk/scratch/tsubame_logs/eval_mdtdist_phase2_v3.qsub.log
#
# TRELLIS v1 Phase 2 v3: full re-clone + build.

source "${COMMON_SH:-/gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/tsubame_jobs/_common.sh}"
tsubame_log_init "eval_mdtdist_phase2_v3"
tsubame_setup_env

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/gs/fs/tga-koike-shanda2/sk/cache/uv
export TORCH_HOME=/gs/fs/tga-koike-shanda2/sk/cache/torch

VENV_PY=/gs/fs/tga-koike-shanda2/sk/envs/mdtdist_uv/bin/python
REPO_DIR=/gs/fs/tga-koike-shanda2/sk/models/trellis_v1
PIP_CMD="uv pip install --python $VENV_PY --no-build-isolation"
PIP_REG="uv pip install --python $VENV_PY"

echo "================================================================"
echo "mdtdist Phase 2 v3: $(date -Iseconds)"
nvidia-smi --query-gpu=name --format=csv,noheader -i 0
nvcc --version | head -4
echo "================================================================"
set +e

echo ""
echo "[1/6] Re-clone full repo (submodules included)"
rm -rf $REPO_DIR
git clone --recursive https://github.com/microsoft/TRELLIS $REPO_DIR 2>&1 | tail -8
find $REPO_DIR -maxdepth 5 -name "setup.py" 2>/dev/null | head

echo ""
echo "[2/6] Reinstate torch 2.8 + cu128 + flash_attn"
$PIP_REG --reinstall torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -3
$PIP_REG --reinstall "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.7.4%2Bcu128torch2.8-cp310-cp310-linux_x86_64.whl" 2>&1 | tail -3

echo ""
echo "[3/6] xformers (try wheel that matches torch 2.8)"
$PIP_REG xformers==0.0.32.post2 2>&1 | tail -5

echo ""
echo "[4/6] nvdiffrast (verbose)"
$PIP_REG --verbose "git+https://github.com/NVlabs/nvdiffrast.git" 2>&1 | tail -15

echo ""
echo "[5/6] Build submodule extensions"
for ext in diffoctreerast mip-splatting vox2seq; do
    EXTD=$(find $REPO_DIR -maxdepth 5 -name "$ext" -type d 2>/dev/null | head -1)
    if [ -z "$EXTD" ]; then echo "  $ext: not found"; continue; fi
    if [ -f "$EXTD/setup.py" ]; then
        echo "  building $ext at $EXTD"
        $PIP_CMD -e "$EXTD" 2>&1 | tail -5
    fi
done

echo ""
echo "[6/6] sanity"
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
echo "mdtdist v3 done: $(date -Iseconds)"
echo "================================================================"
