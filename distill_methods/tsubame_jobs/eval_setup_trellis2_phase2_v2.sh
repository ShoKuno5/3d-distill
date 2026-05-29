#!/usr/bin/env bash
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=3:00:00
#$ -N trellis2-phase2-v2
#$ -j y
#$ -o /gs/fs/tga-koike-shanda2/sk/scratch/tsubame_logs/eval_trellis2_phase2_v2.qsub.log
#
# TRELLIS.2 Phase 2 v2: fix submodule + nvdiffrast issues.
#   - Re-init recursive submodules (depth-1 clone missed CuMesh/FlexGEMM/vox2seq)
#   - Force-install nvdiffrast with --no-build-isolation + verbose to see why it failed
#   - Verify CUDA extensions

source "${COMMON_SH:-/gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/tsubame_jobs/_common.sh}"
tsubame_log_init "eval_trellis2_phase2_v2"
tsubame_setup_env

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/gs/fs/tga-koike-shanda2/sk/cache/uv
export TORCH_HOME=/gs/fs/tga-koike-shanda2/sk/cache/torch

VENV_PY=/gs/fs/tga-koike-shanda2/sk/envs/trellis2_uv/bin/python
REPO_DIR=/gs/fs/tga-koike-shanda2/sk/models/trellis2
PIP_CMD="uv pip install --python $VENV_PY --no-build-isolation"
PIP_CMD_REG="uv pip install --python $VENV_PY"

echo "================================================================"
echo "TRELLIS.2 Phase 2 v2: $(date -Iseconds), host=$(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader -i 0
nvcc --version | head -4
echo "================================================================"
set +e

echo ""
echo "[1/4] Re-init recursive submodules"
cd $REPO_DIR
git submodule update --init --recursive 2>&1 | tail -10
find . -maxdepth 4 -name "setup.py" | head

echo ""
echo "[2/4] Install nvdiffrast (verbose)"
$PIP_CMD_REG "git+https://github.com/NVlabs/nvdiffrast.git@v0.4.0" 2>&1 | tail -15

echo ""
echo "[3/4] Try CuMesh / FlexGEMM / vox2seq builds"
for ext in CuMesh FlexGEMM vox2seq; do
    EXTD=$(find $REPO_DIR -maxdepth 5 -name "$ext" -type d 2>/dev/null | head -1)
    if [ -z "$EXTD" ]; then echo "  $ext: still not found"; continue; fi
    if [ -f "$EXTD/setup.py" ] || [ -f "$EXTD/pyproject.toml" ]; then
        echo "  building $ext at $EXTD"
        $PIP_CMD -e "$EXTD" 2>&1 | tail -8
    else
        echo "  $ext: no setup at $EXTD"
    fi
done

echo ""
echo "[4/4] Sanity"
$VENV_PY -c "
import torch
print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())
for mod in ['nvdiffrast', 'trimesh', 'transformers', 'flash_attn']:
    try:
        m = __import__(mod); v = getattr(m, '__version__', '?')
        print(f'  {mod}: {v}')
    except Exception as e:
        print(f'  {mod}: FAIL {e}')
"
echo "================================================================"
echo "trellis2 v2 done: $(date -Iseconds)"
echo "================================================================"
