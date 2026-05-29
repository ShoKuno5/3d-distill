#!/usr/bin/env bash
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=3:00:00
#$ -N trellis2-env-v4
#$ -j y
#$ -o /gs/fs/tga-koike-shanda2/sk/scratch/tsubame_logs/setup_trellis2_env_v4.qsub.log
#
# trellis2_uv env recovery v4 (after numpy 1.26 fix).
# Builds: nvdiffrast (v0.4.0), cumesh, flexgemm, o-voxel (in-repo), nvdiffrec.
# All clones/builds live under /gs/fs/.../sk/extensions/ (shanda2), never login /tmp.

source "${COMMON_SH:-/gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/tsubame_jobs/_common.sh}"
tsubame_log_init "setup_trellis2_env_v4"
tsubame_setup_env

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/gs/fs/tga-koike-shanda2/sk/cache/uv
export TORCH_HOME=/gs/fs/tga-koike-shanda2/sk/cache/torch
export UV_CONCURRENT_DOWNLOADS=1
export UV_CONCURRENT_BUILDS=1
export UV_CONCURRENT_INSTALLS=1

VENV_PY=/gs/fs/tga-koike-shanda2/sk/envs/trellis2_uv/bin/python
EXT_ROOT=/gs/fs/tga-koike-shanda2/sk/extensions
PIP_NOISO="uv pip install --python $VENV_PY --no-build-isolation"
PIP_REG="uv pip install --python $VENV_PY"

mkdir -p "$EXT_ROOT"

echo "================================================================"
echo "trellis2 env v4: $(date -Iseconds), host=$(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader -i 0
nvcc --version | head -4
echo "================================================================"

set +e

echo ""
echo "[0/6] sanity: numpy / torch in env"
$VENV_PY -c "
import torch, numpy
print('numpy:', numpy.__version__)
print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())
"

echo ""
echo "[1/6] nvdiffrast v0.4.0"
if [ ! -d "$EXT_ROOT/nvdiffrast" ]; then
    git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git $EXT_ROOT/nvdiffrast 2>&1 | tail -5
fi
$PIP_NOISO $EXT_ROOT/nvdiffrast 2>&1 | tail -15

echo ""
echo "[2/6] cumesh (JeffreyXiang)"
if [ ! -d "$EXT_ROOT/cumesh" ]; then
    git clone --recursive https://github.com/JeffreyXiang/CuMesh.git $EXT_ROOT/cumesh 2>&1 | tail -5
fi
$PIP_NOISO $EXT_ROOT/cumesh 2>&1 | tail -15

echo ""
echo "[3/6] flexgemm (JeffreyXiang)"
if [ ! -d "$EXT_ROOT/flexgemm" ]; then
    git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git $EXT_ROOT/flexgemm 2>&1 | tail -5
fi
$PIP_NOISO $EXT_ROOT/flexgemm 2>&1 | tail -15

echo ""
echo "[4/6] o-voxel (in-repo)"
$PIP_NOISO /gs/fs/tga-koike-shanda2/sk/models/trellis2/o-voxel 2>&1 | tail -15

echo ""
echo "[5/6] nvdiffrec (renderutils branch)"
if [ ! -d "$EXT_ROOT/nvdiffrec" ]; then
    git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git $EXT_ROOT/nvdiffrec 2>&1 | tail -5
fi
$PIP_NOISO $EXT_ROOT/nvdiffrec 2>&1 | tail -15

echo ""
echo "[6/6] sanity: trellis2 import test"
cd /gs/fs/tga-koike-shanda2/sk/models/trellis2
$VENV_PY -c "
import sys; sys.path.insert(0, '.')
try:
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    print('Trellis2ImageTo3DPipeline: OK')
except Exception as e:
    import traceback; traceback.print_exc()
    print('Trellis2 import FAIL:', e)
" 2>&1

echo ""
echo "[summary] installed packages (key ones)"
$VENV_PY -c "
for mod in ['numpy','torch','nvdiffrast','cumesh','flexgemm','o_voxel','nvdiffrec','trimesh','flash_attn','xformers']:
    try:
        m = __import__(mod); v = getattr(m, '__version__', '?')
        print(f'  {mod}: {v}')
    except Exception as e:
        print(f'  {mod}: FAIL {e}')
"

echo "================================================================"
echo "trellis2 env v4 done: $(date -Iseconds)"
echo "================================================================"
