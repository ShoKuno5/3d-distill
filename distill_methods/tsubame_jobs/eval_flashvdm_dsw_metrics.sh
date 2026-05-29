#!/usr/bin/env bash
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=1:30:00
#$ -N flashvdm-eval
#$ -j y
#$ -o /gs/fs/tga-koike-shanda2/sk/scratch/tsubame_logs/eval_flashvdm_dsw_metrics.qsub.log
#
# flashvdm_dsw eval-only: run_eval.py で Tier1+2 metric 計算
# 前提: predictions/flashvdm_dsw/default/ に 105 meshes 揃ってる

source "${COMMON_SH:-/gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/tsubame_jobs/_common.sh}"
tsubame_log_init "eval_flashvdm_dsw_metrics"
tsubame_setup_env

REPO_EVAL=/gs/fs/tga-koike-shanda2/sk/3d-distill-eval
CFG=$REPO_EVAL/distill_methods/config_flashvdm_dsw_eval.yaml

echo "================================================================"
echo "flashvdm_dsw eval-only: $(date -Iseconds), host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader -i 0
echo "config: $CFG"
echo "================================================================"

M2_RUN=/gs/fs/tga-koike-shanda2/sk/scratch/distill_methods/m2_cd_tsubame_20260519_1629
N_PRED=$(find "$M2_RUN/predictions/flashvdm_dsw" -name "mesh_raw.obj" 2>/dev/null | wc -l)
echo "flashvdm_dsw predictions: $N_PRED"

cd "$REPO_EVAL"
export PYTHONPATH="$REPO_EVAL/pipeline:$REPO_EVAL/distill_methods:$PYTHONPATH"

CUDA_VISIBLE_DEVICES=0 "$PY" \
    "$REPO_EVAL/pipeline/scripts/run_eval.py" \
    --config "$CFG" \
    --models flashvdm_dsw \
    --workers 4

RC=$?
echo "run_eval.py exit: $RC"
echo "================================================================"
echo "flashvdm_dsw eval-only done: $(date -Iseconds)"
echo "================================================================"
