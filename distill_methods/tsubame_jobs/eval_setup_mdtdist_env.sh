#!/usr/bin/env bash
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=2:00:00
#$ -N eval-mdtdist-env
#$ -j y
#$ -o /gs/fs/tga-koike-shanda2/sk/scratch/tsubame_logs/eval_setup_mdtdist_env.qsub.log
#
# Day 5 prep (Phase 1): TRELLIS v1 (base for mdt_dist) env basic setup.
#   - uv venv with Python 3.10
#   - clone microsoft/TRELLIS (v1)
#   - install Python deps + torch 2.8 cu128
#   - install flash-attn 2.7.4 prebuilt
#   - install spconv-cu120 (CUDA 12.x prebuilt)
#   - skip xformers, kaolin, nvdiffrast (Phase 2 if inference fails)
#   - sanity: import basic modules
#
# mdt_dist weights themselves loaded later via inference script.

source "${COMMON_SH:-/gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/tsubame_jobs/_common.sh}"
tsubame_log_init "eval_setup_mdtdist_env"
tsubame_setup_env

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/gs/fs/tga-koike-shanda2/sk/cache/uv  # home quota avoidance
export TORCH_HOME=/gs/fs/tga-koike-shanda2/sk/cache/torch

ENVS_ROOT=/gs/fs/tga-koike-shanda2/sk/envs
REPOS_ROOT=/gs/fs/tga-koike-shanda2/sk/models
VENV_DIR=$ENVS_ROOT/mdtdist_uv
REPO_DIR=$REPOS_ROOT/trellis_v1

echo "================================================================"
echo "mdt_dist (TRELLIS v1) env setup (Phase 1): $(date -Iseconds), host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader -i 0
nvcc --version 2>&1 | head -4
uv --version
echo "================================================================"

# ----- 1. Create uv venv -----
echo ""
echo "[1/6] Create uv venv with Python 3.10"
mkdir -p $ENVS_ROOT
if [ -d $VENV_DIR ]; then
    echo "  exists: $VENV_DIR (skipping create)"
else
    uv venv --python 3.10 $VENV_DIR
fi
VENV_PY=$VENV_DIR/bin/python
echo "  venv python: $VENV_PY"
$VENV_PY --version

PIP_CMD="uv pip install --python $VENV_PY"

# ----- 2. Clone TRELLIS v1 -----
echo ""
echo "[2/6] Clone microsoft/TRELLIS (v1)"
mkdir -p $REPOS_ROOT
if [ -d $REPO_DIR ]; then
    echo "  exists: $REPO_DIR (skipping clone)"
else
    git clone --recursive --depth 1 https://github.com/microsoft/TRELLIS $REPO_DIR
fi

# ----- 3. Install torch -----
echo ""
echo "[3/6] Install torch 2.8 cu128"
$PIP_CMD torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -3

# ----- 4. Install basic Python deps (TRELLIS v1 list) -----
echo ""
echo "[4/6] Install Python deps"
$PIP_CMD \
    pillow imageio imageio-ffmpeg tqdm easydict \
    opencv-python-headless scipy ninja rembg \
    onnxruntime trimesh open3d xatlas \
    pyvista pymeshfix igraph \
    transformers \
    tensorboard pandas lpips \
    "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8" \
    2>&1 | tail -5

# ----- 5. Install flash-attn + spconv -----
echo ""
echo "[5/6] Install flash-attn prebuilt + spconv-cu120"
$PIP_CMD "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.7.4%2Bcu128torch2.8-cp310-cp310-linux_x86_64.whl" 2>&1 | tail -3
# spconv-cu120 is the closest prebuilt for CUDA 12.x family
$PIP_CMD spconv-cu120 2>&1 | tail -3

# ----- 6. Sanity check -----
echo ""
echo "[6/6] Sanity imports"
$VENV_PY -c "
import torch
print('  torch:', torch.__version__, 'cuda:', torch.cuda.is_available())
import transformers, trimesh
print('  transformers:', transformers.__version__)
print('  trimesh:', trimesh.__version__)
try:
    import flash_attn
    print('  flash_attn:', flash_attn.__version__)
except Exception as e:
    print('  flash_attn FAIL:', e)
try:
    import spconv
    print('  spconv:', spconv.__version__)
except Exception as e:
    print('  spconv FAIL:', e)
"

echo ""
echo "=== Note ==="
echo "Phase 1 complete (basic Python deps + flash-attn + spconv)."
echo "MDT-Dist weights (separate from TRELLIS v1) need to be obtained."
echo "Phase 2 (xformers, kaolin, nvdiffrast etc.) deferred."
echo "================================================================"
echo "mdtdist env setup done: $(date -Iseconds)"
echo "================================================================"
