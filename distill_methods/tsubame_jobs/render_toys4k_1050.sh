#!/usr/bin/env bash
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=1:00:00
#$ -N render-1050
#$ -j y
#$ -o /gs/fs/tga-koike-shanda2/sk/scratch/tsubame_logs/render_toys4k_1050.qsub.log
#
# Render 945 missing toys4k 1050 input images via Blender 3.6 (Cycles + OPTIX).
# 4 GPU intra-node sharded. Submit multiple times with SHARD_OFFSET to scale.
#   qsub -ar 7083 -g tga-koike-shanda -v NUM_SHARDS=16,SHARD_OFFSET=0 render_toys4k_1050.sh
#   qsub -ar 7083 -g tga-koike-shanda -v NUM_SHARDS=16,SHARD_OFFSET=4 render_toys4k_1050.sh
#   qsub -ar 7083 -g tga-koike-shanda -v NUM_SHARDS=16,SHARD_OFFSET=8 render_toys4k_1050.sh
#   qsub -ar 7083 -g tga-koike-shanda -v NUM_SHARDS=16,SHARD_OFFSET=12 render_toys4k_1050.sh

source "${COMMON_SH:-/gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/tsubame_jobs/_common.sh}"
tsubame_log_init "render_toys4k_1050"
tsubame_setup_env

export BLENDER_BIN=/gs/fs/tga-koike-shanda2/sk/blender-3.6.18-linux-x64/blender

REPO=/gs/fs/tga-koike-shanda2/sk/3d-distill
MANIFEST=$REPO/distill_methods/manifests/toys4k_eval_1000/test.csv
SHARD_LOG_DIR=/gs/fs/tga-koike-shanda2/sk/scratch/render_logs/toys4k_1050
NUM_SHARDS=${NUM_SHARDS:-4}
SHARD_OFFSET=${SHARD_OFFSET:-0}

mkdir -p "$SHARD_LOG_DIR"

echo "================================================================"
echo "render toys4k 1050: $(date -Iseconds), host=$(hostname)"
nvidia-smi -L
echo "NUM_SHARDS=$NUM_SHARDS, SHARD_OFFSET=$SHARD_OFFSET"
echo "================================================================"

cd "$REPO"

PIDS=()
for K in 0 1 2 3; do
    SHARD_IDX=$((K + SHARD_OFFSET))
    LOG="$SHARD_LOG_DIR/shard_${SHARD_IDX}.log"
    echo "[launch] GPU $K shard $SHARD_IDX/$NUM_SHARDS -> $LOG"
    CUDA_VISIBLE_DEVICES=$K "$PY" \
        distill_methods/scripts/render_manifest.py \
        --manifest "$MANIFEST" \
        --resolution 512 \
        --timeout-sec 300 \
        --shard "$SHARD_IDX" \
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
RENDERED=$(find /gs/fs/tga-koike-shanda2/sk/data/toys4k/renders/512 -name 'image.png' 2>/dev/null | wc -l)
echo "total renders now: $RENDERED / 1050"

echo "================================================================"
echo "render done: $(date -Iseconds), rc=$EXIT_RC"
echo "================================================================"
exit $EXIT_RC
