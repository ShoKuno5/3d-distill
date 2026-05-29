#!/usr/bin/env bash
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=0:30:00
#$ -N mdt-spconv2
#$ -j y
#$ -o /gs/fs/tga-koike-shanda2/sk/scratch/tsubame_logs/smoke_mdtdist_spconv2.qsub.log
#
# Smoke: retry fries_006 / tree_012 with SPCONV_ALGO=auto.

source "${COMMON_SH:-/gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/tsubame_jobs/_common.sh}"
tsubame_log_init "smoke_mdtdist_spconv2"
tsubame_setup_env

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/gs/fs/tga-koike-shanda2/sk/cache/uv
export TORCH_HOME=/gs/fs/tga-koike-shanda2/sk/cache/torch
export HF_HOME=/gs/fs/tga-koike-shanda2/sk/hf_cache
export SPCONV_ALGO=auto   # ← the key change
export TORCH_CUDA_ARCH_LIST="9.0"

VENV_PY=/gs/fs/tga-koike-shanda2/sk/envs/mdtdist_uv/bin/python
REPO_DIR=/gs/fs/tga-koike-shanda2/sk/models/trellis_v1
INF_SCRIPT=/gs/fs/tga-koike-shanda2/sk/3d-distill/pipeline/scripts/run_inference_mdt_dist.py
CFG=/gs/fs/tga-koike-shanda2/sk/3d-distill-eval/distill_methods/config_mdt_dist_smoke_spconv2.yaml

echo "================================================================"
echo "SPCONV_ALGO=auto smoke: $(date -Iseconds), host=$(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader -i 0
echo "SPCONV_ALGO=$SPCONV_ALGO"
echo "================================================================"

export PYTHONPATH="$REPO_DIR:/gs/fs/tga-koike-shanda2/sk/3d-distill/pipeline:$PYTHONPATH"
cd "$REPO_DIR"

CUDA_VISIBLE_DEVICES=0 $VENV_PY $INF_SCRIPT --config $CFG 2>&1 | tail -60

echo ""
echo "===== result ====="
for oid in fries_006 tree_012; do
    f=/gs/fs/tga-koike-shanda2/sk/scratch/distill_methods/m2_cd_tsubame_20260519_1629/eval_1050/predictions/mdt_dist/default/$oid/mesh_raw.obj
    if [ -f "$f" ]; then
        SIZE=$(stat -c %s "$f")
        echo "  $oid: OK ($SIZE bytes)"
    else
        echo "  $oid: MISSING"
    fi
done

echo "================================================================"
echo "smoke done: $(date -Iseconds)"
echo "================================================================"
