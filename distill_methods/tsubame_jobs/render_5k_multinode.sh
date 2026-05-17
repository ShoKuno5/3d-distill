#!/usr/bin/env bash
#$ -cwd
#$ -l node_f=3
#$ -pe openmpi 3
#$ -l h_rt=2:00:00
#$ -N render-5k-multinode
#$ -j y
#$ -o /gs/fs/tga-koike-shanda2/sk/scratch/tsubame_logs/render_5k_multinode.qsub.log
#
# Render the M2 5K balanced manifest in parallel across 4 nodes x
# 4 GPU = 16 shards. Render is embarrassingly parallel (Blender
# subprocess per mesh, one CUDA device each), so we reuse the MPI
# bootstrap from _common.sh just to fan out 1 bash per node; each
# node then spawns 4 local Python renderers.
#
# Estimate: 5000 / 16 = ~313 rows per shard, ~12 s per render =>
# ~63 min wall clock. h_rt=2h leaves a safety margin.
#
# Submit:  qsub -ar 6925 -g tga-koike-shanda render_5k_multinode.sh

source "${COMMON_SH:-/gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/tsubame_jobs/_common.sh}"
tsubame_log_init "render_5k_multinode"
tsubame_setup_env
tsubame_setup_multinode_env

export BLENDER_BIN="$TSUBAME_ROOT/blender-3.6.18-linux-x64/blender"
export MANIFEST="$REPO/distill_methods/manifests/5k_balanced/train.csv"

echo "================================================================"
echo "Render 5K multinode: $(date -Iseconds)"
echo "host: $(hostname)  NNODES=$NNODES NPROC_PER_NODE=$NPROC_PER_NODE"
echo "manifest: $MANIFEST"
echo "================================================================"

if [ -n "${PE_HOSTFILE:-}" ] && [ "$NNODES" -gt 1 ]; then
    OMPI_HOSTFILE=$(tsubame_make_ompi_hostfile)
    trap 'rm -f "$OMPI_HOSTFILE"' EXIT

    export TOTAL_SHARDS=$((NNODES * NPROC_PER_NODE))
    echo "Total shards = NNODES ($NNODES) * NPROC_PER_NODE ($NPROC_PER_NODE) = $TOTAL_SHARDS"

    # One mpirun task per node. Inside, fan out NPROC_PER_NODE Blender
    # renderers for shards [node_rank * 4 .. node_rank * 4 + 3] out of
    # TOTAL_SHARDS.
    mpirun -n "$NNODES" --map-by ppr:1:node --hostfile "$OMPI_HOSTFILE" \
        -x PATH -x LD_LIBRARY_PATH \
        -x BLENDER_BIN -x TSUBAME_ROOT -x REPO -x VENV -x PY \
        -x MANIFEST -x TOTAL_SHARDS -x NPROC_PER_NODE \
        bash -c "
            NODE_RANK=\${OMPI_COMM_WORLD_RANK:-0}
            echo \"[host \$(hostname)] node_rank=\$NODE_RANK total_shards=\$TOTAL_SHARDS\"
            PIDS=()
            for L in \$(seq 0 \$((NPROC_PER_NODE - 1))); do
                SHARD=\$((NODE_RANK * NPROC_PER_NODE + L))
                LOG=\"\$TSUBAME_ROOT/scratch/tsubame_logs/render_5k_shard\${SHARD}.log\"
                CUDA_VISIBLE_DEVICES=\$L \\
                    \"\$PY\" \"\$REPO/distill_methods/scripts/render_manifest.py\" \\
                    --manifest \"\$MANIFEST\" \\
                    --resolution 512 \\
                    --timeout-sec 300 \\
                    --shard \$SHARD --num-shards \$TOTAL_SHARDS \\
                    > \"\$LOG\" 2>&1 &
                PIDS+=(\$!)
                echo \"[host \$(hostname)] launched shard \$SHARD on GPU \$L (pid \${PIDS[-1]})\"
            done
            for pid in \${PIDS[@]}; do wait \$pid; done
            echo \"[host \$(hostname)] all local shards finished\"
        "
else
    echo "ERROR: multi-node PE allocation missing, refusing to fall back"
    exit 1
fi

echo "================================================================"
echo "Render 5K multinode done: $(date -Iseconds)"
RENDERED=$(find /gs/fs/tga-koike-shanda2/sk/data/5k_balanced -name image.png 2>/dev/null | wc -l)
echo "rendered count: $RENDERED / 5000"
echo "================================================================"
