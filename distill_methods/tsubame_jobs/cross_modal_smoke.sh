#!/usr/bin/env bash
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=0:30:00
#$ -N xmodal-smoke
#$ -j y
#$ -o /gs/fs/tga-koike-shanda2/sk/scratch/tsubame_logs/cross_modal_smoke.qsub.log
#
# Cross-modal smoke: 3 samples x 1 model on 1 GPU. Validates that ULIP + Uni3D
# load together (namespace-isolation fix), termcolor is present, models fail
# FAST if anything is missing, and ULIP-I/Uni3D-I scores actually come out.
# Writes to a throwaway CSV so it never touches the real cross_modal.csv.
#
# Submit: qsub -ar 7083 -g tga-koike-shanda cross_modal_smoke.sh

source "${COMMON_SH:-/gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/tsubame_jobs/_common.sh}"
tsubame_log_init "cross_modal_smoke"
tsubame_setup_env

REPO_EVAL=/gs/fs/tga-koike-shanda2/sk/3d-distill-eval
CFG=$REPO_EVAL/distill_methods/config_tsubame_1050_eval.yaml
OUT=/gs/fs/tga-koike-shanda2/sk/scratch/tsubame_logs/smoke_cross_modal.csv

echo "================================================================"
echo "cross-modal smoke: $(date -Iseconds), host=$(hostname)"
nvidia-smi -L | head -1
echo "================================================================"

cd "$REPO_EVAL"
export PYTHONPATH="$REPO_EVAL/pipeline:$REPO_EVAL/distill_methods:$PYTHONPATH"

CUDA_VISIBLE_DEVICES=0 "$PY" \
    "$REPO_EVAL/pipeline/scripts/run_cross_modal.py" \
    --config "$CFG" \
    --models teacher_50step \
    --max-samples 3 \
    --num-shards 1 --shard-index 0 \
    --out-csv "$OUT"
RC=$?

echo "===== smoke cross_modal.csv ====="
cat "$OUT" 2>/dev/null
echo "================================================================"
echo "cross-modal smoke done: $(date -Iseconds), rc=$RC"
echo "================================================================"
exit $RC
