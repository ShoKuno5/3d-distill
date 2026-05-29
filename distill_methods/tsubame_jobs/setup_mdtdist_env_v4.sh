#!/usr/bin/env bash
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=3:00:00
#$ -N mdtdist-env-v4
#$ -j y
#$ -o /gs/fs/tga-koike-shanda2/sk/scratch/tsubame_logs/setup_mdtdist_env_v4.qsub.log
#
# mdtdist_uv env recovery v4 (after numpy 1.26 fix).
# Builds for TRELLIS v1 / mdt-dist: nvdiffrast, diffoctreerast, mip-splatting
# (diff-gaussian-rasterization), vox2seq (in-repo). All under shanda2.

source "${COMMON_SH:-/gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/tsubame_jobs/_common.sh}"
tsubame_log_init "setup_mdtdist_env_v4"
tsubame_setup_env

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/gs/fs/tga-koike-shanda2/sk/cache/uv
export TORCH_HOME=/gs/fs/tga-koike-shanda2/sk/cache/torch
export UV_CONCURRENT_DOWNLOADS=1
export UV_CONCURRENT_BUILDS=1
export UV_CONCURRENT_INSTALLS=1

VENV_PY=/gs/fs/tga-koike-shanda2/sk/envs/mdtdist_uv/bin/python
EXT_ROOT=/gs/fs/tga-koike-shanda2/sk/extensions
TRELLIS_V1=/gs/fs/tga-koike-shanda2/sk/models/trellis_v1
PIP_NOISO="uv pip install --python $VENV_PY --no-build-isolation"

mkdir -p "$EXT_ROOT"

echo "================================================================"
echo "mdtdist env v4: $(date -Iseconds), host=$(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader -i 0
nvcc --version | head -4
echo "================================================================"

set +e

echo ""
echo "[0/5] sanity: numpy / torch"
$VENV_PY -c "
import torch, numpy
print('numpy:', numpy.__version__)
print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())
"

echo ""
echo "[1/5] nvdiffrast (reuse from trellis2 build if exists)"
if [ ! -d "$EXT_ROOT/nvdiffrast" ]; then
    git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git $EXT_ROOT/nvdiffrast 2>&1 | tail -5
fi
$PIP_NOISO $EXT_ROOT/nvdiffrast 2>&1 | tail -10

echo ""
echo "[2/5] diffoctreerast (JeffreyXiang, --recurse-submodules)"
if [ ! -d "$EXT_ROOT/diffoctreerast" ]; then
    git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git $EXT_ROOT/diffoctreerast 2>&1 | tail -5
fi
$PIP_NOISO $EXT_ROOT/diffoctreerast 2>&1 | tail -10

echo ""
echo "[3/5] mip-splatting / diff-gaussian-rasterization"
if [ ! -d "$EXT_ROOT/mip-splatting" ]; then
    git clone --recursive https://github.com/autonomousvision/mip-splatting.git $EXT_ROOT/mip-splatting 2>&1 | tail -5
fi
$PIP_NOISO $EXT_ROOT/mip-splatting/submodules/diff-gaussian-rasterization 2>&1 | tail -10

echo ""
echo "[4/5] vox2seq (in trellis_v1 repo)"
VOX2SEQ_PATH=$(find $TRELLIS_V1 -maxdepth 6 -type d -name "vox2seq" 2>/dev/null | head -1)
if [ -z "$VOX2SEQ_PATH" ]; then
    echo "  vox2seq not in TRELLIS v1 clone; check trellis_v1/trellis/modules/"
    find $TRELLIS_V1/trellis/modules -maxdepth 4 -type d 2>/dev/null | head -20
else
    echo "  vox2seq at: $VOX2SEQ_PATH"
    $PIP_NOISO $VOX2SEQ_PATH 2>&1 | tail -10
fi

echo ""
echo "[5/5] sanity: TRELLIS v1 import"
cd $TRELLIS_V1
$VENV_PY -c "
import sys; sys.path.insert(0, '.')
import os; os.environ['SPCONV_ALGO'] = 'native'
try:
    from trellis.pipelines import TrellisImageTo3DPipeline
    print('TrellisImageTo3DPipeline: OK')
except Exception as e:
    import traceback; traceback.print_exc()
    print('TRELLIS v1 import FAIL:', e)
" 2>&1

echo ""
echo "[summary] key packages"
$VENV_PY -c "
for mod in ['numpy','torch','nvdiffrast','diffoctreerast','diff_gaussian_rasterization','vox2seq','trimesh','flash_attn','spconv']:
    try:
        m = __import__(mod); v = getattr(m, '__version__', '?')
        print(f'  {mod}: {v}')
    except Exception as e:
        print(f'  {mod}: FAIL {e}')
"

echo "================================================================"
echo "mdtdist env v4 done: $(date -Iseconds)"
echo "================================================================"
