#!/usr/bin/env bash
#$ -cwd
#$ -l h_rt=120:00:00
#$ -l node_f=2
#$ -pe openmpi 2
#$ -N m2dmd2-multinode
#$ -j y
#$ -o /gs/fs/tga-koike-shanda2/sk/scratch/tsubame_logs/m2_dmd2_multinode.qsub.log
#
# M2 DMD2 distillation on TSUBAME 4.0 H100, multi-node DDP.
#   - 2 nodes x 4 GPU = 8 H100 SXM5 96GB
#     (capped at 8 GPU per user as fair team share even on ar 6991+6992)
#   - Per-GPU batch = 4, effective batch = 32
#   - 5K-sample training data (HSSD + Objaverse-XL balanced mix;
#     ABO deferred until tar extraction)
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
# Resume from a checkpoint (re-uses the original RUN_NAME / OUTPUT_ROOT):
#   qsub -ar <id> -g tga-koike-shanda \
#     -v SRC_CFG_PATH=...,RESUME_CKPT=/gs/fs/.../m2_dmd2_multinode_20260519_1022/checkpoints/dmd2/step_10000 \
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

if [ -n "${RESUME_CKPT:-}" ]; then
    if [ ! -d "$RESUME_CKPT" ]; then
        echo "RESUME_CKPT does not exist: $RESUME_CKPT" >&2
        exit 1
    fi
    # $RESUME_CKPT layout: <OUTPUT_ROOT>/checkpoints/dmd2/step_XXXXX
    OUTPUT_ROOT="$(cd "$RESUME_CKPT/../../.." && pwd)"
    RUN_NAME="$(basename "$OUTPUT_ROOT")"
    RESUME_ARG=(--resume "$RESUME_CKPT")
    echo "[resume] from $RESUME_CKPT"
    echo "[resume] reusing OUTPUT_ROOT=$OUTPUT_ROOT"
else
    RUN_NAME="m2_dmd2_multinode_$(date +%Y%m%d_%H%M)"
    OUTPUT_ROOT="$TSUBAME_ROOT/scratch/distill_methods/$RUN_NAME"
    RESUME_ARG=()
fi
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
    # Inline RESUME_CKPT (if set) into the remote bash; quotes are safe
    # because the path lives under $TSUBAME_ROOT (no spaces).
    RESUME_INLINE=""
    if [ -n "${RESUME_CKPT:-}" ]; then
        RESUME_INLINE="--resume $RESUME_CKPT"
    fi
    mpirun -n "$NNODES" --map-by ppr:1:node --hostfile "$OMPI_HOSTFILE" \
        -x PATH -x LD_LIBRARY_PATH \
        -x MASTER_ADDR -x MASTER_PORT \
        bash -c "
            source $REPO/distill_methods/tsubame_jobs/_common.sh
            tsubame_setup_env
            tsubame_setup_multinode_env
            tsubame_torchrun_multinode \
                $REPO/distill_methods/src/train.py \
                --config $RUN_CFG --method dmd2 \
                --output-dir $OUTPUT_ROOT \
                $RESUME_INLINE
        "
else
    # Single-node fallback (e.g. testing with node_f=1).
    "$TORCHRUN" --nproc_per_node="$NPROC_PER_NODE" \
        "$REPO/distill_methods/src/train.py" \
        --config "$RUN_CFG" --method dmd2 \
        --output-dir "$OUTPUT_ROOT" \
        "${RESUME_ARG[@]}"
fi

echo "================================================================"
echo "M2 DMD2 done: $(date -Iseconds)"
echo "checkpoints: $OUTPUT_ROOT/checkpoints/dmd2/"
echo "================================================================"
