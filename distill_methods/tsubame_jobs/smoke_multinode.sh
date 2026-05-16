#!/usr/bin/env bash
#$ -cwd
#$ -l node_f=2
#$ -pe openmpi 2
#$ -l h_rt=1:00:00
#$ -N smoke-multinode
#$ -j y
#$ -o /gs/fs/tga-koike-shanda2/sk/scratch/tsubame_logs/smoke_multinode.qsub.log
#
# Multi-node DDP smoke: 2 nodes x 4 GPU = 8 ranks, ~10 step DMD2.
# Verifies:
#   - UGE openmpi PE allocation + PE_HOSTFILE parsing
#   - MASTER_ADDR / NODE_RANK propagation through mpirun
#   - NCCL InfiniBand transport between nodes
#   - smoke_test_ddp_dmd2.py PASS H1 (discriminator cross-rank sync) +
#     PASS H2 (replay-buffer indices) under 8-rank
#   - DDP fix (no_sync + manual _sync_grads) is rank-count agnostic
#
# Submit example:
#   qsub -ar 6925 -g tga-koike-shanda smoke_multinode.sh

source "${COMMON_SH:-/gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/tsubame_jobs/_common.sh}"
tsubame_log_init "smoke_multinode"
tsubame_setup_env
tsubame_setup_multinode_env

cd "$REPO/models/hunyuan3d21/hy3dshape"

echo "================================================================"
echo "MULTINODE SMOKE: $(date -Iseconds)"
echo "host: $(hostname)"
echo "PE_HOSTFILE contents:"
[ -n "${PE_HOSTFILE:-}" ] && cat "$PE_HOSTFILE" || echo "  (no PE_HOSTFILE)"
echo "NNODES=$NNODES MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "================================================================"

if [ -n "${PE_HOSTFILE:-}" ] && [ "$NNODES" -gt 1 ]; then
    # OpenMPI 5 PRTE can't parse UGE's 4-column PE_HOSTFILE; convert.
    OMPI_HOSTFILE=$(tsubame_make_ompi_hostfile)
    trap 'rm -f "$OMPI_HOSTFILE"' EXIT
    # --map-by ppr:1:node = one task per node
    # bash -c (NOT -lc): -l re-sources /etc/profile and may unset
    # OMPI_COMM_WORLD_RANK / PMIX_RANK that mpirun set per-task.
    # -x forwards specific env vars explicitly (defensive).
    mpirun -n "$NNODES" --map-by ppr:1:node --hostfile "$OMPI_HOSTFILE" \
        -x PATH -x LD_LIBRARY_PATH \
        -x MASTER_ADDR -x MASTER_PORT \
        bash -c "
            echo \"[host \$(hostname)] BEFORE setup: OMPI_COMM_WORLD_RANK=\${OMPI_COMM_WORLD_RANK:-unset} PMIX_RANK=\${PMIX_RANK:-unset}\"
            env | grep -E 'OMPI|PMIX|PMI_' | head -10
            source $REPO/distill_methods/tsubame_jobs/_common.sh
            tsubame_setup_env
            tsubame_setup_multinode_env
            echo \"[host \$(hostname)] AFTER setup: node_rank=\$NODE_RANK -> launching torchrun on \$NPROC_PER_NODE GPU\"
            tsubame_torchrun_multinode \
                $REPO/distill_methods/scripts/smoke_test_ddp_dmd2.py \
                --config $REPO/distill_methods/configs/config_525_hssd_tsubame.yaml \
                --steps 10
        "
else
    echo "[warn] no multi-node PE allocation; running single-node smoke"
    "$TORCHRUN" --nproc_per_node="$NPROC_PER_NODE" \
        --master_port="$MASTER_PORT" \
        "$REPO/distill_methods/scripts/smoke_test_ddp_dmd2.py" \
        --config "$REPO/distill_methods/configs/config_525_hssd_tsubame.yaml" \
        --steps 10
fi

echo "================================================================"
echo "SMOKE done: $(date -Iseconds) exit=$?"
echo "================================================================"
