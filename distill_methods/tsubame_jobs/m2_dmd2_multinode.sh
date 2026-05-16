#!/usr/bin/env bash
#$ -cwd
#$ -l node_f=4
#$ -pe openmpi 4
#$ -l h_rt=24:00:00
#$ -N m2dmd2-multinode
#$ -j y
#$ -o /gs/fs/tga-koike-shanda2/sk/scratch/tsubame_logs/m2_dmd2_multinode.qsub.log
#
# M2 DMD2 distillation on TSUBAME 4.0 H100, multi-node DDP.
#   - 4 nodes x 4 GPU = 16 H100 SXM5 96GB
#   - Per-GPU batch = 4, effective batch = 64
#   - 5K-sample training data (HSSD + ABO + Objaverse-LVIS balanced mix)
#   - 15000 steps, gradient checkpointing on
#
# Layout assumption:
#   $SRC_CFG_PATH points at the M2 config (5K manifest). Set NNODES if
#   you want to override the PE-derived value.
#
# Submit example:
#   qsub -ar 6991 -g tga-koike-shanda \
#     -v SRC_CFG_PATH=/gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/configs/config_5k_m2_tsubame.yaml \
#     m2_dmd2_multinode.sh
#
# Mechanism: UGE openmpi PE allocates N node slots; mpirun launches one
# bash per node; each bash calls tsubame_torchrun_multinode which fans
# out to the 4 GPUs on that node. NCCL all-reduce is intra-node NVLink
# + inter-node Mellanox HDR200 IB. The DDP fix from commits
# 5c7fbb8 / 4eee4fd / 15f2bed (no_sync + manual _sync_grads) is
# rank-count agnostic and works the same across 4 nodes.

source "${COMMON_SH:-/gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/tsubame_jobs/_common.sh}"
tsubame_log_init "m2_dmd2_multinode"
tsubame_setup_env
tsubame_setup_multinode_env

RUN_NAME="m2_dmd2_multinode_$(date +%Y%m%d_%H%M)"
OUTPUT_ROOT="$TSUBAME_ROOT/scratch/distill_methods/$RUN_NAME"
RUN_CFG="$OUTPUT_ROOT/config.yaml"

cd "$REPO/models/hunyuan3d21/hy3dshape"
tsubame_print_banner "M2 DMD2 multi-node" "$RUN_NAME" "$OUTPUT_ROOT"
tsubame_make_run_config "$OUTPUT_ROOT" "$RUN_CFG"

# Each PE slot (one per node) executes this branch. mpirun fans out via
# OMPI_COMM_WORLD_RANK so tsubame_setup_multinode_env above already set
# NODE_RANK correctly on each node.
if [ -n "${PE_HOSTFILE:-}" ] && [ "$NNODES" -gt 1 ]; then
    # OpenMPI 5 PRTE can't parse UGE's 4-column PE_HOSTFILE; convert.
    OMPI_HOSTFILE=$(tsubame_make_ompi_hostfile)
    trap 'rm -f "$OMPI_HOSTFILE"' EXIT
    # Multi-node path: mpirun is the outer launcher, this script is the
    # per-node entrypoint. Each node runs torchrun for its 4 local GPUs.
    # OpenMPI 5: --map-by ppr:1:node = one task per node
    mpirun -n "$NNODES" --map-by ppr:1:node --hostfile "$OMPI_HOSTFILE" \
        bash -lc "
            source $REPO/distill_methods/tsubame_jobs/_common.sh
            tsubame_setup_env
            tsubame_setup_multinode_env
            tsubame_torchrun_multinode \
                $REPO/distill_methods/src/train.py \
                --config $RUN_CFG --method dmd2 \
                --output-dir $OUTPUT_ROOT
        "
else
    # Single-node fallback (e.g. testing with node_f=1).
    "$TORCHRUN" --nproc_per_node="$NPROC_PER_NODE" \
        "$REPO/distill_methods/src/train.py" \
        --config "$RUN_CFG" --method dmd2 \
        --output-dir "$OUTPUT_ROOT"
fi

echo "================================================================"
echo "M2 DMD2 done: $(date -Iseconds)"
echo "checkpoints: $OUTPUT_ROOT/checkpoints/dmd2/"
echo "================================================================"
