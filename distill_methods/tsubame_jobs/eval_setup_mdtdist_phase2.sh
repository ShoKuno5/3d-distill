#!/usr/bin/env bash
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=4:00:00
#$ -N mdtdist-phase2
#$ -j y
#$ -o /gs/fs/tga-koike-shanda2/sk/scratch/tsubame_logs/eval_mdtdist_phase2.qsub.log
#
# TRELLIS v1 (base for mdt_dist) Phase 2: heavier CUDA extensions + weights.
#   - xformers (matching torch 2.8)
#   - kaolin (NVIDIA wheel)
#   - nvdiffrast, diffoctreerast, mip-splatting (submodules)
#   - vox2seq (local extension)
#   - DL microsoft/TRELLIS-image-large weights
# MDT-Dist specific weights (Zhou et al. 2025) need separate sourcing — TBD.

source "${COMMON_SH:-/gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/tsubame_jobs/_common.sh}"
tsubame_log_init "eval_mdtdist_phase2"
tsubame_setup_env

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/gs/fs/tga-koike-shanda2/sk/cache/uv
export TORCH_HOME=/gs/fs/tga-koike-shanda2/sk/cache/torch

VENV_DIR=/gs/fs/tga-koike-shanda2/sk/envs/mdtdist_uv
VENV_PY=$VENV_DIR/bin/python
REPO_DIR=/gs/fs/tga-koike-shanda2/sk/models/trellis_v1
PIP_CMD="uv pip install --python $VENV_PY --no-build-isolation"
PIP_CMD_REG="uv pip install --python $VENV_PY"

echo "================================================================"
echo "TRELLIS v1 (mdt_dist) Phase 2: $(date -Iseconds), host=$(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader -i 0
nvcc --version | head -4
echo "================================================================"

set +e

# ----- 1. xformers -----
echo ""
echo "[1/6] Install xformers (matching torch 2.8 + cu128)"
$PIP_CMD_REG xformers 2>&1 | tail -5

# ----- 2. nvdiffrast -----
echo ""
echo "[2/6] nvdiffrast"
$PIP_CMD_REG "git+https://github.com/NVlabs/nvdiffrast.git" 2>&1 | tail -5

# ----- 3. diffoctreerast + mip-splatting (submodules) -----
echo ""
echo "[3/6] diffoctreerast + mip-splatting"
for ext in diffoctreerast mip-splatting; do
    EXTD=$(find $REPO_DIR -maxdepth 4 -name "$ext" -type d 2>/dev/null | head -1)
    if [ -z "$EXTD" ]; then echo "  $ext: not found in repo, skipping"; continue; fi
    echo "  building $ext at $EXTD"
    $PIP_CMD -e "$EXTD" 2>&1 | tail -5
done

# ----- 4. vox2seq (local extension in TRELLIS v1) -----
echo ""
echo "[4/6] vox2seq"
VOX2SEQ=$(find $REPO_DIR -maxdepth 4 -name "vox2seq" -type d 2>/dev/null | head -1)
if [ -n "$VOX2SEQ" ]; then
    echo "  building vox2seq at $VOX2SEQ"
    $PIP_CMD -e "$VOX2SEQ" 2>&1 | tail -5
else
    echo "  vox2seq not found"
fi

# ----- 5. TRELLIS-image-large weights -----
echo ""
echo "[5/6] DL microsoft/TRELLIS-image-large weights"
$VENV_PY -c "
from huggingface_hub import snapshot_download
try:
    p = snapshot_download(repo_id='microsoft/TRELLIS-image-large')
    print('cache root:', p)
except Exception as e:
    print('DL fail:', e)
" 2>&1 | head -5

# ----- 6. Sanity -----
echo ""
echo "[6/6] Sanity imports"
$VENV_PY -c "
import torch
print('  torch:', torch.__version__, 'cuda:', torch.cuda.is_available())
for mod in ['xformers', 'nvdiffrast', 'spconv', 'flash_attn', 'trimesh', 'transformers']:
    try:
        m = __import__(mod)
        v = getattr(m, '__version__', '?')
        print(f'  {mod}: {v}')
    except Exception as e:
        print(f'  {mod}: FAIL {e}')
"

echo ""
echo "=== Note ==="
echo "TRELLIS v1 env Phase 2 done (best-effort)."
echo "MDT-Dist weights (separate from TRELLIS v1 base) need manual location."
echo "================================================================"
echo "mdtdist Phase 2 done: $(date -Iseconds)"
echo "================================================================"
