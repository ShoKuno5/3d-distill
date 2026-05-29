#!/usr/bin/env bash
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=2:00:00
#$ -N trellis2-inference
#$ -j y
#$ -o /gs/fs/tga-koike-shanda2/sk/scratch/tsubame_logs/eval_trellis2_inference.qsub.log
#
# TRELLIS.2 inference on Toys4k 105. Uses trellis2_uv venv + TRELLIS.2 repo at
# /gs/fs/.../sk/models/trellis2. Output: predictions/trellis2/default/<oid>/mesh.obj

source "${COMMON_SH:-/gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/tsubame_jobs/_common.sh}"
tsubame_log_init "eval_trellis2_inference"
tsubame_setup_env

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/gs/fs/tga-koike-shanda2/sk/cache/uv
export TORCH_HOME=/gs/fs/tga-koike-shanda2/sk/cache/torch
export HF_HOME=/gs/fs/tga-koike-shanda2/sk/hf_cache

VENV_PY=/gs/fs/tga-koike-shanda2/sk/envs/trellis2_uv/bin/python
REPO_DIR=/gs/fs/tga-koike-shanda2/sk/models/trellis2
SCRIPT=/gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/scripts/trellis2_inference.py

echo "================================================================"
echo "TRELLIS.2 inference: $(date -Iseconds), host=$(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader -i 0
echo "================================================================"

# Add TRELLIS.2 repo to PYTHONPATH for imports
export PYTHONPATH="$REPO_DIR:$PYTHONPATH"
cd $REPO_DIR

$VENV_PY $SCRIPT

echo "================================================================"
echo "trellis2 inference done: $(date -Iseconds)"
M2=/gs/fs/tga-koike-shanda2/sk/scratch/distill_methods/m2_cd_tsubame_20260519_1629
echo "predictions: $(find $M2/predictions/trellis2 -name 'mesh.obj' 2>/dev/null | wc -l) / 105"
echo "================================================================"
