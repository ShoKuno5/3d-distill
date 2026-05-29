#!/usr/bin/env bash
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=2:00:00
#$ -N eval-trellis2-env
#$ -j y
#$ -o /gs/fs/tga-koike-shanda2/sk/scratch/tsubame_logs/eval_setup_trellis2_env.qsub.log
#
# Day 5 prep (Phase 1): TRELLIS.2 env basic setup.
#   - uv venv with Python 3.10
#   - clone microsoft/TRELLIS.2
#   - install Python deps (torch 2.8 + cu128, transformers, timm, etc.)
#   - install flash-attn 2.7.4 prebuilt for cu128+torch2.8+cp310
#   - skip heavy CUDA extensions (nvdiffrast, CuMesh, FlexGEMM) → Phase 2
#   - sanity: import basic modules
#
# Continue-on-error: log issues but don't abort the whole script.

source "${COMMON_SH:-/gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/tsubame_jobs/_common.sh}"
tsubame_log_init "eval_setup_trellis2_env"
tsubame_setup_env

# Use uv (already in $HOME/.local/bin)
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/gs/fs/tga-koike-shanda2/sk/cache/uv  # home quota avoidance
export TORCH_HOME=/gs/fs/tga-koike-shanda2/sk/cache/torch

ENVS_ROOT=/gs/fs/tga-koike-shanda2/sk/envs
REPOS_ROOT=/gs/fs/tga-koike-shanda2/sk/models
VENV_DIR=$ENVS_ROOT/trellis2_uv
REPO_DIR=$REPOS_ROOT/trellis2

echo "================================================================"
echo "trellis2 env setup (Phase 1): $(date -Iseconds), host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader -i 0
nvcc --version 2>&1 | head -4
uv --version
echo "================================================================"

# ----- 1. Create uv venv -----
echo ""
echo "[1/5] Create uv venv with Python 3.10"
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

# ----- 2. Clone TRELLIS.2 -----
echo ""
echo "[2/5] Clone microsoft/TRELLIS.2"
mkdir -p $REPOS_ROOT
if [ -d $REPO_DIR ]; then
    echo "  exists: $REPO_DIR (skipping clone)"
else
    git clone --recursive --depth 1 https://github.com/microsoft/TRELLIS.2 $REPO_DIR
fi

# ----- 3. Install torch -----
echo ""
echo "[3/5] Install torch 2.8 cu128"
$PIP_CMD torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -3

# ----- 4. Install basic Python deps -----
echo ""
echo "[4/5] Install Python deps"
$PIP_CMD \
    transformers timm kornia tqdm easydict \
    imageio imageio-ffmpeg pillow \
    opencv-python-headless ninja trimesh \
    tensorboard pandas lpips zstandard \
    "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8" \
    2>&1 | tail -5

# Flash-attn prebuilt (matching torch 2.8 + cu128 + cp310)
echo ""
echo "[4b/5] Install flash-attn prebuilt"
$PIP_CMD "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.7.4%2Bcu128torch2.8-cp310-cp310-linux_x86_64.whl" 2>&1 | tail -3

# ----- 5. Sanity check (basic imports only, skip CUDA extensions) -----
echo ""
echo "[5/5] Sanity imports"
$VENV_PY -c "
import torch
print('  torch:', torch.__version__, 'cuda:', torch.cuda.is_available())
import transformers, timm, trimesh, kornia
print('  transformers:', transformers.__version__)
print('  timm:', timm.__version__)
print('  trimesh:', trimesh.__version__)
try:
    import flash_attn
    print('  flash_attn:', flash_attn.__version__)
except Exception as e:
    print('  flash_attn FAIL:', e)
"

echo ""
echo "=== Note ==="
echo "Phase 1 complete (basic Python deps + flash-attn)."
echo "Phase 2 (heavy CUDA extensions like nvdiffrast, CuMesh, FlexGEMM)"
echo "will be needed for actual inference, deferred until eval phase."
echo "================================================================"
echo "trellis2 env setup done: $(date -Iseconds)"
echo "================================================================"
