#!/usr/bin/env bash
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=4:00:00
#$ -N trellis2-phase2
#$ -j y
#$ -o /gs/fs/tga-koike-shanda2/sk/scratch/tsubame_logs/eval_trellis2_phase2.qsub.log
#
# TRELLIS.2 Phase 2: CUDA extensions + model weights DL.
#   - flash-attn (already in Phase 1)
#   - nvdiffrast v0.4.0 from source
#   - nvdiffrec renderutils branch
#   - CuMesh, FlexGEMM submodules
#   - DL microsoft/TRELLIS.2-4B weights
# Defensive: log failures, continue past errors so we know what works.

source "${COMMON_SH:-/gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/tsubame_jobs/_common.sh}"
tsubame_log_init "eval_trellis2_phase2"
tsubame_setup_env

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/gs/fs/tga-koike-shanda2/sk/cache/uv
export TORCH_HOME=/gs/fs/tga-koike-shanda2/sk/cache/torch

VENV_DIR=/gs/fs/tga-koike-shanda2/sk/envs/trellis2_uv
VENV_PY=$VENV_DIR/bin/python
REPO_DIR=/gs/fs/tga-koike-shanda2/sk/models/trellis2
PIP_CMD="uv pip install --python $VENV_PY --no-build-isolation"
PIP_CMD_REG="uv pip install --python $VENV_PY"

echo "================================================================"
echo "TRELLIS.2 Phase 2: $(date -Iseconds), host=$(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader -i 0
nvcc --version | head -4
echo "================================================================"

set +e  # continue on errors

# ----- 1. nvdiffrast -----
echo ""
echo "[1/5] Install nvdiffrast v0.4.0"
if $VENV_PY -c "import nvdiffrast" 2>/dev/null; then echo "  exists"; else
    $PIP_CMD_REG "git+https://github.com/NVlabs/nvdiffrast.git@v0.4.0" 2>&1 | tail -8
fi

# ----- 2. CuMesh + FlexGEMM (submodules of TRELLIS.2) -----
echo ""
echo "[2/5] Build CuMesh + FlexGEMM"
for ext in CuMesh FlexGEMM; do
    EXTD=$(find $REPO_DIR -maxdepth 3 -name "$ext" -type d 2>/dev/null | head -1)
    if [ -z "$EXTD" ]; then echo "  $ext: not found in repo, skipping"; continue; fi
    if [ ! -f "$EXTD/setup.py" ] && [ ! -f "$EXTD/pyproject.toml" ]; then
        echo "  $ext: no setup.py/pyproject in $EXTD, skipping"
        continue
    fi
    echo "  building $ext at $EXTD"
    $PIP_CMD -e "$EXTD" 2>&1 | tail -5
done

# ----- 3. utils3d (already in Phase 1, verify) -----
echo ""
echo "[3/5] utils3d verify"
$VENV_PY -c "import utils3d; print('utils3d OK')" 2>&1 | head -3

# ----- 4. TRELLIS.2 weights -----
echo ""
echo "[4/5] DL microsoft/TRELLIS.2-4B weights"
$VENV_PY -c "
from huggingface_hub import snapshot_download
try:
    p = snapshot_download(repo_id='microsoft/TRELLIS.2-4B')
    print('cache root:', p)
    import os
    for root, dirs, files in os.walk(p):
        for f in files:
            fp = os.path.join(root, f)
            print(f'  {os.path.relpath(fp, p)}: {os.path.getsize(fp)} bytes')
        if len(dirs) > 0: break
except Exception as e:
    print('DL fail:', e)
" 2>&1 | head -20

# ----- 5. Sanity -----
echo ""
echo "[5/5] Sanity imports"
$VENV_PY -c "
import torch
print('  torch:', torch.__version__, 'cuda:', torch.cuda.is_available())
for mod in ['nvdiffrast', 'utils3d', 'trimesh', 'transformers', 'flash_attn']:
    try:
        m = __import__(mod)
        v = getattr(m, '__version__', '?')
        print(f'  {mod}: {v}')
    except Exception as e:
        print(f'  {mod}: FAIL {e}')
"

echo "================================================================"
echo "TRELLIS.2 Phase 2 done: $(date -Iseconds)"
echo "================================================================"
