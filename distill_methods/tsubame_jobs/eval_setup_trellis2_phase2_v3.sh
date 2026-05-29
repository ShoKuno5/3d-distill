#!/usr/bin/env bash
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=2:00:00
#$ -N trellis2-phase2-v3
#$ -j y
#$ -o /gs/fs/tga-koike-shanda2/sk/scratch/tsubame_logs/eval_trellis2_phase2_v3.qsub.log
#
# TRELLIS.2 Phase 2 v3: full re-clone + build.
#   - rm -rf old depth-1 clone (submodule refs broken)
#   - git clone --recursive --no-depth-limit microsoft/TRELLIS.2
#   - nvdiffrast install with verbose log
#   - build CuMesh / FlexGEMM / vox2seq if found

source "${COMMON_SH:-/gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/tsubame_jobs/_common.sh}"
tsubame_log_init "eval_trellis2_phase2_v3"
tsubame_setup_env

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/gs/fs/tga-koike-shanda2/sk/cache/uv
export TORCH_HOME=/gs/fs/tga-koike-shanda2/sk/cache/torch

VENV_PY=/gs/fs/tga-koike-shanda2/sk/envs/trellis2_uv/bin/python
REPO_DIR=/gs/fs/tga-koike-shanda2/sk/models/trellis2
PIP_CMD="uv pip install --python $VENV_PY --no-build-isolation"
PIP_REG="uv pip install --python $VENV_PY"

echo "================================================================"
echo "TRELLIS.2 Phase 2 v3 (full re-clone): $(date -Iseconds)"
nvidia-smi --query-gpu=name --format=csv,noheader -i 0
nvcc --version | head -4
echo "================================================================"
set +e

echo ""
echo "[1/5] rm old + full clone with submodules"
rm -rf $REPO_DIR
git clone --recursive https://github.com/microsoft/TRELLIS.2 $REPO_DIR 2>&1 | tail -8
find $REPO_DIR -maxdepth 3 -name "setup.py" 2>/dev/null | head

echo ""
echo "[2/5] nvdiffrast (verbose)"
$PIP_REG --verbose "git+https://github.com/NVlabs/nvdiffrast.git@v0.4.0" 2>&1 | tail -20

echo ""
echo "[3/5] Submodule extension builds"
for ext in CuMesh FlexGEMM vox2seq nvdiffrec; do
    EXTD=$(find $REPO_DIR -maxdepth 5 -name "$ext" -type d 2>/dev/null | head -1)
    if [ -z "$EXTD" ]; then echo "  $ext: not found"; continue; fi
    if [ -f "$EXTD/setup.py" ] || [ -f "$EXTD/pyproject.toml" ]; then
        echo "  building $ext at $EXTD"
        $PIP_CMD -e "$EXTD" 2>&1 | tail -5
    fi
done

echo ""
echo "[4/5] sanity"
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

echo ""
echo "[5/5] check TRELLIS.2 imports"
cd $REPO_DIR
$VENV_PY -c "
import sys; sys.path.insert(0, '.')
try:
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    print('  Trellis2ImageTo3DPipeline OK')
except Exception as e:
    print(f'  Trellis2 import FAIL: {e}')
"

echo "================================================================"
echo "trellis2 v3 done: $(date -Iseconds)"
echo "================================================================"
