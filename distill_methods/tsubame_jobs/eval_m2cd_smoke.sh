#!/usr/bin/env bash
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=3:00:00
#$ -N m2cd-smoke
#$ -j y
#$ -o /gs/fs/tga-koike-shanda2/sk/scratch/tsubame_logs/eval_m2cd_smoke.qsub.log
#
# M2 CD smoke (Day 1-A 仕上げ):
#   Phase 1: cd_4step inference on Toys4k 105 sample
#   Phase 2: run_eval.py (CD/F-score/Hausdorff/FD-Inception/FD-DINOv2)
# 1 GPU, h_rt=3h (inference ~15min + eval ~1.5h想定)

source "${COMMON_SH:-/gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/tsubame_jobs/_common.sh}"
tsubame_log_init "eval_m2cd_smoke"
tsubame_setup_env

REPO_MAIN=/gs/fs/tga-koike-shanda2/sk/3d-distill
REPO_EVAL=/gs/fs/tga-koike-shanda2/sk/3d-distill-eval
CFG=$REPO_EVAL/distill_methods/config_tsubame_cd_eval.yaml

echo "================================================================"
echo "M2 CD smoke: $(date -Iseconds), host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader -i 0
echo "config: $CFG"
echo "================================================================"

# -------- Phase 1: Inference (cd_4step) --------
echo ""
echo "===== Phase 1: cd_4step inference ====="
cd "$REPO_MAIN/models/hunyuan3d21/hy3dshape"
export PYTHONPATH="$REPO_MAIN/models/hunyuan3d21/hy3dshape:$REPO_EVAL/pipeline:$REPO_EVAL/distill_methods:$PYTHONPATH"

CUDA_VISIBLE_DEVICES=0 "$PY" \
    "$REPO_EVAL/distill_methods/scripts/run_inference_distilled.py" \
    --config "$CFG" \
    --model-name cd_4step

PHASE1_RC=$?
echo "Phase 1 exit: $PHASE1_RC"
if [ "$PHASE1_RC" != "0" ]; then
    echo "ERROR: inference failed, aborting"
    exit $PHASE1_RC
fi

# -------- Phase 2: Eval (geometry + FD) --------
echo ""
echo "===== Phase 2: run_eval.py (cd_4step only) ====="
cd "$REPO_EVAL"
export PYTHONPATH="$REPO_EVAL/pipeline:$REPO_EVAL/distill_methods:$PYTHONPATH"

CUDA_VISIBLE_DEVICES=0 "$PY" \
    "$REPO_EVAL/pipeline/scripts/run_eval.py" \
    --config "$CFG" \
    --models cd_4step \
    --workers 4

PHASE2_RC=$?
echo "Phase 2 exit: $PHASE2_RC"

# -------- Summary --------
echo ""
echo "===== Summary ====="
M2_RUN=/gs/fs/tga-koike-shanda2/sk/scratch/distill_methods/m2_cd_tsubame_20260519_1629
echo "predictions:"
find "$M2_RUN/predictions/cd_4step" -name "mesh_raw.obj" 2>/dev/null | wc -l
echo "metrics CSVs:"
ls -la "$M2_RUN/metrics/" 2>/dev/null
echo ""
echo "frechet_distance.csv contents:"
cat "$M2_RUN/metrics/frechet_distance.csv" 2>/dev/null
echo ""
echo "summary.csv (cd_4step row):"
grep -E "^model|cd_4step" "$M2_RUN/metrics/summary.csv" 2>/dev/null | head

echo "================================================================"
echo "M2 CD smoke done: $(date -Iseconds)"
echo "================================================================"
