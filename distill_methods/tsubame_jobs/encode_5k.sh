#!/usr/bin/env bash
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=24:00:00
#$ -N encode-5k
#$ -j y
#$ -o /gs/fs/tga-koike-shanda2/sk/scratch/tsubame_logs/encode_5k.qsub.log
#
# Pre-encode VAE latents for the M2 5K balanced manifest in parallel
# across 4 GPU on one node_f. Uses prepare_training_data.py's built-in
# --shard / --num-shards.
#
# Estimate: 5000 / 4 = 1250 rows per shard, ~30 s per teacher 50-step
# ODE encode => ~10.4 h per shard, 4 shards in parallel => ~10.4 h
# wall clock. Set h_rt to 24h for safety margin.
#
# Submit:  qsub -ar 6925 -g tga-koike-shanda encode_5k.sh

source "${COMMON_SH:-/gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/tsubame_jobs/_common.sh}"
tsubame_log_init "encode_5k"
tsubame_setup_env

CFG="$REPO/distill_methods/configs/config_5k_m2_tsubame.yaml"
NPROC=$(nvidia-smi -L | wc -l)
cd "$REPO/models/hunyuan3d21/hy3dshape"
echo "================================================================"
echo "Encode 5K balanced: $(date -Iseconds)"
echo "host: $(hostname)  gpu count: $NPROC"
echo "config: $CFG"
echo "================================================================"

PIDS=()
for SHARD in $(seq 0 $((NPROC - 1))); do
    LOG="$TSUBAME_ROOT/scratch/tsubame_logs/encode_5k_shard${SHARD}.log"
    CUDA_VISIBLE_DEVICES=$SHARD \
    "$PY" "$REPO/distill_methods/scripts/prepare_training_data.py" \
        --config "$CFG" \
        --shard "$SHARD" --num-shards "$NPROC" \
        > "$LOG" 2>&1 &
    PIDS+=($!)
    echo "started shard $SHARD on GPU $SHARD (pid ${PIDS[-1]}, log $LOG)"
done

echo "================================================================"
echo "Waiting for $NPROC encoding shards to finish ..."
for pid in "${PIDS[@]}"; do
    wait "$pid"
done

echo "================================================================"
echo "Encode 5K done: $(date -Iseconds)"
ENCODED=$(ls /gs/fs/tga-koike-shanda2/sk/data/5k_balanced/training_data/*.npz 2>/dev/null | wc -l)
echo "encoded count: $ENCODED / 5000"
echo "================================================================"
