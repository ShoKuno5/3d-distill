#!/usr/bin/env bash
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=0:30:00
#$ -N mdtdist-smoke
#$ -j y
#$ -o /gs/fs/tga-koike-shanda2/sk/scratch/tsubame_logs/smoke_mdtdist.qsub.log
#
# MDT-Dist smoke test: TRELLIS v1 import + 1-step inference on 5 samples.

source "${COMMON_SH:-/gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/tsubame_jobs/_common.sh}"
tsubame_log_init "smoke_mdtdist"
tsubame_setup_env

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/gs/fs/tga-koike-shanda2/sk/cache/uv
export TORCH_HOME=/gs/fs/tga-koike-shanda2/sk/cache/torch
export HF_HOME=/gs/fs/tga-koike-shanda2/sk/hf_cache
export SPCONV_ALGO=native
export TORCH_CUDA_ARCH_LIST="9.0"

VENV_PY=/gs/fs/tga-koike-shanda2/sk/envs/mdtdist_uv/bin/python
REPO_DIR=/gs/fs/tga-koike-shanda2/sk/models/trellis_v1
INF_SCRIPT=/gs/fs/tga-koike-shanda2/sk/3d-distill/pipeline/scripts/run_inference_mdt_dist.py
CFG=/gs/fs/tga-koike-shanda2/sk/3d-distill-eval/distill_methods/config_mdt_dist_1050.yaml

echo "================================================================"
echo "MDT-Dist smoke: $(date -Iseconds), host=$(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader -i 0
echo "================================================================"

echo ""
echo "[1] kaolin C-ext import (was the ABI mismatch)"
$VENV_PY -c "import kaolin; print('kaolin:', kaolin.__version__); from kaolin import _C; print('  _C import OK')" 2>&1 | tail -5

echo ""
echo "[2] TRELLIS v1 pipeline import"
cd "$REPO_DIR"
$VENV_PY -c "
import sys; sys.path.insert(0, '.')
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.pipelines.samplers import FlowEulerGuidanceIntervalSampler
print('TrellisImageTo3DPipeline / FlowEulerGuidanceIntervalSampler import OK')
" 2>&1 | tail -5

echo ""
echo "[3] inference on 5 samples"
export PYTHONPATH="$REPO_DIR:/gs/fs/tga-koike-shanda2/sk/3d-distill/pipeline:$PYTHONPATH"
CUDA_VISIBLE_DEVICES=0 $VENV_PY $INF_SCRIPT --config $CFG --max-samples 5 2>&1 | tail -30

echo ""
echo "===== smoke result ====="
N=$(find /gs/fs/tga-koike-shanda2/sk/scratch/distill_methods/m2_cd_tsubame_20260519_1629/eval_1050/predictions/mdt_dist -name "mesh_raw.obj" 2>/dev/null | wc -l)
echo "predictions: $N"

echo "================================================================"
echo "smoke done: $(date -Iseconds)"
echo "================================================================"
