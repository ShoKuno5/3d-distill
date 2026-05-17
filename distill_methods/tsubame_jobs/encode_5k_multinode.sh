#!/usr/bin/env bash
#$ -cwd
#$ -l node_f=4
#$ -pe openmpi 4
#$ -l h_rt=6:00:00
#$ -N encode-5k-multinode
#$ -j y
#$ -o /gs/fs/tga-koike-shanda2/sk/scratch/tsubame_logs/encode_5k_multinode.qsub.log
#
# Pre-encode VAE latents for the M2 5K balanced manifest across 4
# nodes x 4 GPU = 16 shards using prepare_training_data.py's built-in
# --shard / --num-shards. Each shard runs the teacher 50-step ODE on
# ~313 samples.
#
# Estimate: 5000 / 16 = ~313 rows per shard, ~30 s per encode =>
# ~157 min wall clock. h_rt=6h leaves margin (first-run model load
# adds ~3 min cold start per shard).
#
# Submit:  qsub -ar 6925 -g tga-koike-shanda encode_5k_multinode.sh

source "${COMMON_SH:-/gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/tsubame_jobs/_common.sh}"
tsubame_log_init "encode_5k_multinode"
tsubame_setup_env
tsubame_setup_multinode_env

export CFG="$REPO/distill_methods/configs/config_5k_m2_tsubame.yaml"
cd "$REPO/models/hunyuan3d21/hy3dshape"
echo "================================================================"
echo "Encode 5K multinode: $(date -Iseconds)"
echo "host: $(hostname)  NNODES=$NNODES NPROC_PER_NODE=$NPROC_PER_NODE"
echo "config: $CFG"
echo "================================================================"

if [ -n "${PE_HOSTFILE:-}" ] && [ "$NNODES" -gt 1 ]; then
    OMPI_HOSTFILE=$(tsubame_make_ompi_hostfile)
    trap 'rm -f "$OMPI_HOSTFILE"' EXIT

    export TOTAL_SHARDS=$((NNODES * NPROC_PER_NODE))
    echo "Total shards = $TOTAL_SHARDS"

    mpirun -n "$NNODES" --map-by ppr:1:node --hostfile "$OMPI_HOSTFILE" \
        -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH \
        -x HF_HOME -x WANDB_API_KEY -x HF_TOKEN \
        -x TSUBAME_ROOT -x REPO -x VENV -x PY \
        -x CFG -x TOTAL_SHARDS -x NPROC_PER_NODE \
        bash -c "
            NODE_RANK=\${OMPI_COMM_WORLD_RANK:-0}
            cd \"\$REPO/models/hunyuan3d21/hy3dshape\"
            echo \"[host \$(hostname)] node_rank=\$NODE_RANK total_shards=\$TOTAL_SHARDS\"
            PIDS=()
            for L in \$(seq 0 \$((NPROC_PER_NODE - 1))); do
                SHARD=\$((NODE_RANK * NPROC_PER_NODE + L))
                LOG=\"\$TSUBAME_ROOT/scratch/tsubame_logs/encode_5k_shard\${SHARD}.log\"
                CUDA_VISIBLE_DEVICES=\$L \\
                    \"\$PY\" \"\$REPO/distill_methods/scripts/prepare_training_data.py\" \\
                    --config \"\$CFG\" \\
                    --shard \$SHARD --num-shards \$TOTAL_SHARDS \\
                    > \"\$LOG\" 2>&1 &
                PIDS+=(\$!)
                echo \"[host \$(hostname)] launched shard \$SHARD on GPU \$L (pid \${PIDS[-1]})\"
            done
            for pid in \${PIDS[@]}; do wait \$pid; done
            echo \"[host \$(hostname)] all local encoding shards finished\"
        "
else
    echo "ERROR: multi-node PE allocation missing"
    exit 1
fi

echo "================================================================"
echo "Encode 5K multinode done: $(date -Iseconds)"
ENCODED=$(ls /gs/fs/tga-koike-shanda2/sk/data/5k_balanced/training_data/*.npz 2>/dev/null | wc -l)
echo "encoded count: $ENCODED / 5000"
echo "================================================================"
