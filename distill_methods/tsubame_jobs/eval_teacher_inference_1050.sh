#!/usr/bin/env bash
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=6:00:00
#$ -N teacher-1050
#$ -j y
#$ -o /gs/fs/tga-koike-shanda2/sk/scratch/tsubame_logs/eval_teacher_inference_1050.qsub.log
#
# teacher_50step (HY3D-2.1) inference on Toys4k 1050 with intra-node sharding.
# 4 GPU per job. Use env vars to span multiple jobs:
#   - NUM_SHARDS:    total shards across all jobs   (default 4)
#   - SHARD_OFFSET:  this job's first shard index   (default 0)
# Examples:
#   # Single node_f (4 GPU, 4 shards):
#   qsub -ar 7083 -g tga-koike-shanda eval_teacher_inference_1050.sh
#   # Two node_f (8 GPU, 8 shards), submit twice:
#   qsub -ar 7083 -g tga-koike-shanda -v NUM_SHARDS=8,SHARD_OFFSET=0 eval_teacher_inference_1050.sh
#   qsub -ar 7083 -g tga-koike-shanda -v NUM_SHARDS=8,SHARD_OFFSET=4 eval_teacher_inference_1050.sh

source "${COMMON_SH:-/gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/tsubame_jobs/_common.sh}"
tsubame_log_init "eval_teacher_inference_1050"
tsubame_setup_env
export HY3DGEN_MODELS=/gs/fs/tga-koike-shanda2/sk/cache/hy3dgen

REPO_MAIN=/gs/fs/tga-koike-shanda2/sk/3d-distill
REPO_EVAL=/gs/fs/tga-koike-shanda2/sk/3d-distill-eval
CFG=$REPO_EVAL/distill_methods/config_tsubame_1050_eval.yaml
EVAL_ROOT=/gs/fs/tga-koike-shanda2/sk/scratch/distill_methods/m2_cd_tsubame_20260519_1629/eval_1050
SHARD_LOG_DIR=$EVAL_ROOT/inference_logs/teacher_50step
NUM_SHARDS=${NUM_SHARDS:-4}
SHARD_OFFSET=${SHARD_OFFSET:-0}

mkdir -p "$EVAL_ROOT" "$SHARD_LOG_DIR"

echo "================================================================"
echo "teacher_50step 1050 inference: $(date -Iseconds), host=$(hostname)"
nvidia-smi -L
echo "config: $CFG"
echo "NUM_SHARDS=$NUM_SHARDS, SHARD_OFFSET=$SHARD_OFFSET"
echo "================================================================"

cd "$REPO_MAIN/models/hunyuan3d21/hy3dshape"
export PYTHONPATH="$REPO_MAIN/models/hunyuan3d21/hy3dshape:$REPO_EVAL/pipeline:$REPO_EVAL/distill_methods:$PYTHONPATH"

PIDS=()
for K in 0 1 2 3; do
    SHARD_IDX=$((K + SHARD_OFFSET))
    LOG="$SHARD_LOG_DIR/shard_${SHARD_IDX}.log"
    echo "[launch] GPU $K shard $SHARD_IDX/$NUM_SHARDS -> $LOG"
    CUDA_VISIBLE_DEVICES=$K "$PY" \
        "$REPO_EVAL/distill_methods/scripts/run_inference_distilled.py" \
        --config "$CFG" \
        --model-name teacher_50step \
        --shard-index "$SHARD_IDX" \
        --num-shards "$NUM_SHARDS" \
        > "$LOG" 2>&1 &
    PIDS+=($!)
done

EXIT_RC=0
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}"
    rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "[FAIL] shard $((i + SHARD_OFFSET)) exit=$rc"
        EXIT_RC=$rc
    fi
done

echo ""
echo "===== Summary ====="
echo "predictions: $(find $EVAL_ROOT/predictions/teacher_50step -name 'mesh_raw.obj' 2>/dev/null | wc -l) / 1050"
for K in 0 1 2 3; do
    SHARD_IDX=$((K + SHARD_OFFSET))
    echo "--- shard $SHARD_IDX tail ---"
    tail -5 "$SHARD_LOG_DIR/shard_${SHARD_IDX}.log" 2>/dev/null
done

echo "================================================================"
echo "teacher_50step 1050 inference done: $(date -Iseconds), rc=$EXIT_RC"
echo "================================================================"
exit $EXIT_RC
