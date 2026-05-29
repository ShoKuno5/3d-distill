#!/usr/bin/env bash
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=3:00:00
#$ -N render-eval-1050
#$ -j y
#$ -o /gs/fs/tga-koike-shanda2/sk/scratch/tsubame_logs/render_eval_1050.qsub.log
#
# Render eval multiview images (view_<az>.png) for GT + per-model predicted
# meshes of the 1050 cross-family benchmark — the inputs FD and CLIP-I consume.
# (run_eval renders these sequentially in one process, which would time out for
# ~7350 meshes; this shards them across GPUs/nodes.)
#
# 4 GPUs intra-node. Submit 4 times with SHARD_OFFSET 0/4/8/12 to span 4 nodes
# = 16 shards total:
#   qsub -ar 7083 -g tga-koike-shanda -v NUM_SHARDS=16,SHARD_OFFSET=0  render_eval_1050_multinode.sh
#   qsub -ar 7083 -g tga-koike-shanda -v NUM_SHARDS=16,SHARD_OFFSET=4  render_eval_1050_multinode.sh
#   qsub -ar 7083 -g tga-koike-shanda -v NUM_SHARDS=16,SHARD_OFFSET=8  render_eval_1050_multinode.sh
#   qsub -ar 7083 -g tga-koike-shanda -v NUM_SHARDS=16,SHARD_OFFSET=12 render_eval_1050_multinode.sh

source "${COMMON_SH:-/gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/tsubame_jobs/_common.sh}"
tsubame_log_init "render_eval_1050"
tsubame_setup_env

export BLENDER_BIN=/gs/fs/tga-koike-shanda2/sk/blender-3.6.18-linux-x64/blender
REPO_EVAL=/gs/fs/tga-koike-shanda2/sk/3d-distill-eval
CFG=$REPO_EVAL/distill_methods/config_tsubame_1050_eval.yaml
MODELS=${MODELS:-"teacher_50step flashvdm_dsw cd_4step dmd2_1step mdt_dist trellis2"}
NUM_SHARDS=${NUM_SHARDS:-16}
SHARD_OFFSET=${SHARD_OFFSET:-0}
SHARD_LOG_DIR=/gs/fs/tga-koike-shanda2/sk/scratch/render_logs/eval_1050
mkdir -p "$SHARD_LOG_DIR"

echo "================================================================"
echo "render eval 1050: $(date -Iseconds), host=$(hostname)"
nvidia-smi -L
echo "NUM_SHARDS=$NUM_SHARDS SHARD_OFFSET=$SHARD_OFFSET  models=$MODELS"
echo "================================================================"

cd "$REPO_EVAL"
export PYTHONPATH="$REPO_EVAL/pipeline:$REPO_EVAL/distill_methods:$PYTHONPATH"

PIDS=()
for K in 0 1 2 3; do
    SHARD_IDX=$((K + SHARD_OFFSET))
    LOG="$SHARD_LOG_DIR/shard_${SHARD_IDX}.log"
    echo "[launch] GPU $K shard $SHARD_IDX/$NUM_SHARDS -> $LOG"
    CUDA_VISIBLE_DEVICES=$K "$PY" \
        "$REPO_EVAL/pipeline/scripts/render_eval_multiview.py" \
        --config "$CFG" \
        --models $MODELS \
        --shard-index "$SHARD_IDX" --num-shards "$NUM_SHARDS" \
        > "$LOG" 2>&1 &
    PIDS+=($!)
done

EXIT_RC=0
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" || { echo "[FAIL] shard $((i + SHARD_OFFSET)) exit=$?"; EXIT_RC=1; }
done

echo "================================================================"
echo "render eval 1050 done: $(date -Iseconds), rc=$EXIT_RC"
echo "================================================================"
exit $EXIT_RC
