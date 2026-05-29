#!/usr/bin/env bash
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=6:00:00
#$ -N metrics-1050
#$ -j y
#$ -o /gs/fs/tga-koike-shanda2/sk/scratch/tsubame_logs/eval_metrics_1050.qsub.log
#
# Metric computation for 1050 eval (CD/F-score/Hausdorff/FD/CLIP-I/etc).
# Uses node_f (4 GPU + 96 vCPU) with --workers 48 to keep CPU saturated.
# Earlier attempt (job 7766005) used node_q + workers=4 and hit h_rt at 57%.
#
# MODELS env var (space-separated) controls which models to evaluate.
# Default = all 6 models for the cross-family benchmark.

source "${COMMON_SH:-/gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/tsubame_jobs/_common.sh}"
tsubame_log_init "eval_metrics_1050"
tsubame_setup_env

REPO_EVAL=/gs/fs/tga-koike-shanda2/sk/3d-distill-eval
CFG=$REPO_EVAL/distill_methods/config_tsubame_1050_eval.yaml
EVAL_ROOT=/gs/fs/tga-koike-shanda2/sk/scratch/distill_methods/m2_cd_tsubame_20260519_1629/eval_1050
MODELS=${MODELS:-"teacher_50step flashvdm_dsw cd_4step dmd2_1step mdt_dist trellis2"}
WORKERS=${WORKERS:-48}

echo "================================================================"
echo "1050 metrics: $(date -Iseconds), host=$(hostname)"
nvidia-smi -L
echo "config: $CFG"
echo "models: $MODELS"
echo "workers: $WORKERS"
echo "eval_root: $EVAL_ROOT"
echo "================================================================"

# Sanity: count predictions for each requested model
for M in $MODELS; do
    N=$(find "$EVAL_ROOT/predictions/$M" -name 'mesh_raw.obj' -o -name 'mesh.obj' 2>/dev/null | wc -l)
    echo "  $M: $N predictions"
done

cd "$REPO_EVAL"
export PYTHONPATH="$REPO_EVAL/pipeline:$REPO_EVAL/distill_methods:$PYTHONPATH"

"$PY" \
    "$REPO_EVAL/pipeline/scripts/run_eval.py" \
    --config "$CFG" \
    --models $MODELS \
    --workers $WORKERS

RC=$?

echo ""
echo "===== Output ====="
echo "metrics:"
ls -la "$EVAL_ROOT/metrics/" 2>/dev/null
echo ""
echo "summary.csv:"
head -2 "$EVAL_ROOT/metrics/summary.csv" 2>/dev/null
tail -20 "$EVAL_ROOT/metrics/summary.csv" 2>/dev/null

echo "================================================================"
echo "1050 metrics done: $(date -Iseconds), rc=$RC"
echo "================================================================"
exit $RC
