#!/usr/bin/env bash
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=12:00:00
#$ -N render-5k
#$ -j y
#$ -o /gs/fs/tga-koike-shanda2/sk/scratch/tsubame_logs/render_5k.qsub.log
#
# Render the M2 5K balanced manifest in parallel across 4 GPU on one
# node_f. Each shard uses one CUDA device for Blender CYCLES+OPTIX.
#
# Estimate: 5000 / 4 = 1250 rows per shard, ~12 s per render =>
# ~4.2 h per shard, 4 shards in parallel => ~4.2 h wall clock.
#
# Submit:  qsub -ar 6925 -g tga-koike-shanda render_5k.sh

source "${COMMON_SH:-/gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/tsubame_jobs/_common.sh}"
tsubame_log_init "render_5k"
tsubame_setup_env

export BLENDER_BIN="$TSUBAME_ROOT/blender-3.6.18-linux-x64/blender"

MANIFEST="$REPO/distill_methods/manifests/5k_balanced/train.csv"
NPROC=$(nvidia-smi -L | wc -l)
echo "================================================================"
echo "Render 5K balanced: $(date -Iseconds)"
echo "host: $(hostname)  gpu count: $NPROC"
echo "manifest: $MANIFEST"
echo "BLENDER_BIN=$BLENDER_BIN"
echo "================================================================"

PIDS=()
for SHARD in $(seq 0 $((NPROC - 1))); do
    LOG="$TSUBAME_ROOT/scratch/tsubame_logs/render_5k_shard${SHARD}.log"
    CUDA_VISIBLE_DEVICES=$SHARD \
    "$PY" "$REPO/distill_methods/scripts/render_manifest.py" \
        --manifest "$MANIFEST" \
        --resolution 512 \
        --timeout-sec 300 \
        --shard "$SHARD" --num-shards "$NPROC" \
        > "$LOG" 2>&1 &
    PIDS+=($!)
    echo "started shard $SHARD on GPU $SHARD (pid ${PIDS[-1]}, log $LOG)"
done

echo "================================================================"
echo "Waiting for $NPROC shards to finish ..."
for pid in "${PIDS[@]}"; do
    wait "$pid"
done

echo "================================================================"
echo "Render 5K done: $(date -Iseconds)"
RENDERED=$(find /gs/fs/tga-koike-shanda2/sk/data/5k_balanced -name image.png 2>/dev/null | wc -l)
echo "rendered count: $RENDERED / 5000"
echo "================================================================"
