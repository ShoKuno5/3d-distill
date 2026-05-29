#!/usr/bin/env bash
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=2:00:00
#$ -N m2cd-evalonly
#$ -j y
#$ -o /gs/fs/tga-koike-shanda2/sk/scratch/tsubame_logs/eval_m2cd_evalonly.qsub.log
#
# M2 CD eval-only re-run.
# 前提: cd_4step predictions が既に存在 (eval_m2cd_smoke.sh の Phase 1 で生成済)
# 動作: run_eval.py を呼ぶ。inference は skip、render orchestration → FD → CD/F-score 再計算
# 用途: render orchestration patch deploy 後の再 eval、debug 用

source "${COMMON_SH:-/gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/tsubame_jobs/_common.sh}"
tsubame_log_init "eval_m2cd_evalonly"
tsubame_setup_env

REPO_MAIN=/gs/fs/tga-koike-shanda2/sk/3d-distill
REPO_EVAL=/gs/fs/tga-koike-shanda2/sk/3d-distill-eval
CFG=$REPO_EVAL/distill_methods/config_tsubame_cd_eval.yaml

echo "================================================================"
echo "M2 CD eval-only: $(date -Iseconds), host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader -i 0
echo "config: $CFG"
echo "================================================================"

# Sanity check predictions exist
M2_RUN=/gs/fs/tga-koike-shanda2/sk/scratch/distill_methods/m2_cd_tsubame_20260519_1629
N_PRED=$(find "$M2_RUN/predictions/cd_4step" -name "mesh_raw.obj" 2>/dev/null | wc -l)
echo "Existing cd_4step predictions: $N_PRED"
if [ "$N_PRED" -lt 1 ]; then
    echo "ERROR: no predictions found, run eval_m2cd_smoke.sh first"
    exit 1
fi

cd "$REPO_EVAL"
export PYTHONPATH="$REPO_EVAL/pipeline:$REPO_EVAL/distill_methods:$PYTHONPATH"

CUDA_VISIBLE_DEVICES=0 "$PY" \
    "$REPO_EVAL/pipeline/scripts/run_eval.py" \
    --config "$CFG" \
    --models cd_4step teacher_50step flashvdm \
    --workers 4

RC=$?
echo "run_eval.py exit: $RC"

# Summary
echo ""
echo "===== Output ====="
echo "metrics:"
ls -la "$M2_RUN/metrics/" 2>/dev/null
echo ""
echo "multiview_renders (post-render):"
ls "$M2_RUN/multiview_renders/" 2>/dev/null
echo ""
echo "frechet_distance.csv:"
cat "$M2_RUN/metrics/frechet_distance.csv" 2>/dev/null
echo ""
echo "summary.csv (cd_4step rows):"
grep -E "^model|cd_4step" "$M2_RUN/metrics/summary.csv" 2>/dev/null | head

echo "================================================================"
echo "M2 CD eval-only done: $(date -Iseconds)"
echo "================================================================"
