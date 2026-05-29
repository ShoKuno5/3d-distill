#!/usr/bin/env bash
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=0:30:00
#$ -N trellis2-smoke
#$ -j y
#$ -o /gs/fs/tga-koike-shanda2/sk/scratch/tsubame_logs/smoke_trellis2.qsub.log
#
# TRELLIS.2 smoke: end-to-end pipeline load (+DINOv3 dl) + 5 sample inference.

source "${COMMON_SH:-/gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/tsubame_jobs/_common.sh}"
tsubame_log_init "smoke_trellis2"
tsubame_setup_env

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/gs/fs/tga-koike-shanda2/sk/cache/uv
export TORCH_HOME=/gs/fs/tga-koike-shanda2/sk/cache/torch
export HF_HOME=/gs/fs/tga-koike-shanda2/sk/hf_cache

# Tiny manifest of 5 samples
SMOKE_MANIFEST=/gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/manifests/toys4k_smoke_trellis2/test.csv
mkdir -p $(dirname $SMOKE_MANIFEST)
M1050=/gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/manifests/toys4k_eval_1000/test.csv
head -1 $M1050 > $SMOKE_MANIFEST
sed -n '2,6p' $M1050 >> $SMOKE_MANIFEST

VENV_PY=/gs/fs/tga-koike-shanda2/sk/envs/trellis2_uv/bin/python
REPO_DIR=/gs/fs/tga-koike-shanda2/sk/models/trellis2
SCRIPT=/gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/scripts/trellis2_inference.py

# Use a distinct output dir so we don't conflict with the wide run
SMOKE_OUT=/gs/fs/tga-koike-shanda2/sk/scratch/distill_methods/m2_cd_tsubame_20260519_1629/eval_1050/predictions/trellis2_smoke/default
mkdir -p $SMOKE_OUT

echo "================================================================"
echo "TRELLIS.2 smoke: $(date -Iseconds), host=$(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader -i 0
echo "manifest: $SMOKE_MANIFEST"
echo "output: $SMOKE_OUT"
echo "================================================================"

export PYTHONPATH="$REPO_DIR:$PYTHONPATH"
cd $REPO_DIR

CUDA_VISIBLE_DEVICES=0 \
TRELLIS2_MANIFEST="$SMOKE_MANIFEST" \
TRELLIS2_OUTPUT="$SMOKE_OUT" \
$VENV_PY $SCRIPT 2>&1 | tail -60

echo ""
echo "===== result ====="
N=$(find $SMOKE_OUT -name "mesh.obj" 2>/dev/null | wc -l)
echo "predictions: $N / 5"

echo "================================================================"
echo "smoke done: $(date -Iseconds)"
echo "================================================================"
