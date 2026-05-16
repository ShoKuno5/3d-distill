#!/usr/bin/env bash
# Shared setup for TSUBAME 4.0 distill jobs.
#
# Layout assumptions:
#   TSUBAME_ROOT=/gs/fs/tga-koike-shanda2/sk
#     ├── 3d-distill/              (repo clone, this branch: exp/scaling-experiments)
#     │   └── .venv/               (uv venv, Python 3.10)
#     ├── hf_cache/                (HuggingFace cache; tencent/Hunyuan3D-2.1)
#     ├── data/                    (pre-encoded latents + DMD1 pairs + toys4k renders)
#     │   ├── 525_hssd/training_data/*.npz
#     │   ├── 525_hssd/dmd1_pairs/pair_*.npz
#     │   └── toys4k/renders/...
#     └── scratch/distill_methods/ (per-run output)
#
# Usage from a job script:
#   source "$(dirname "$0")/_common.sh"
#   tsubame_setup_env
#   tsubame_log_init m1_pd

set -euo pipefail

TSUBAME_ROOT="${TSUBAME_ROOT:-/gs/fs/tga-koike-shanda2/sk}"
REPO="$TSUBAME_ROOT/3d-distill"
VENV="$REPO/.venv"
PY="$VENV/bin/python"
TORCHRUN="$VENV/bin/torchrun"

tsubame_setup_env() {
    module purge 2>/dev/null || true
    module load cuda/12.8.0 cudnn/9.8.0 2>/dev/null || echo "[warn] module load failed (interactive shell?)"

    export HF_HOME="$TSUBAME_ROOT/hf_cache"
    export PYTHONPATH="$REPO/models/hunyuan3d21/hy3dshape:$REPO/distill_methods/src${PYTHONPATH:+:$PYTHONPATH}"
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    # WANDB online OK on TSUBAME (no egress restriction unlike qzcli)
    if [ -f "$REPO/.env" ]; then
        set -a; . "$REPO/.env"; set +a
    fi
    : "${WANDB_API_KEY:?WANDB_API_KEY must be set (put it in $REPO/.env)}"
}

tsubame_log_init() {
    local NAME="$1"
    local LOGDIR="$TSUBAME_ROOT/scratch/tsubame_logs"
    mkdir -p "$LOGDIR"
    local LOG="$LOGDIR/${NAME}_$(hostname)_$(date +%Y%m%d_%H%M%S).log"
    exec > >(tee -a "$LOG") 2>&1
    echo "[log mirror: $LOG]"
}

tsubame_print_banner() {
    local NAME="$1"
    local RUN_NAME="$2"
    local OUTPUT_ROOT="$3"
    local NGPU
    NGPU=$(nvidia-smi -L | wc -l)
    local GPU_NAME
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    local GPU_MEM
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    echo "================================================================"
    echo "$NAME: $RUN_NAME"
    echo "output: $OUTPUT_ROOT"
    echo "host: $(hostname)  ngpu: $NGPU  gpu: $GPU_NAME ($GPU_MEM)"
    echo "================================================================"
}

# Build a per-run config copy (output_root rewritten, optional overrides).
# Usage: tsubame_make_run_config <RUN_CFG> [extra python override line]
tsubame_make_run_config() {
    local OUTPUT_ROOT="$1"
    local RUN_CFG="$2"
    local EXTRA="${3:-}"
    local SRC_CFG="${SRC_CFG_PATH:-$REPO/distill_methods/configs/config_525_hssd_tsubame.yaml}"
    mkdir -p "$OUTPUT_ROOT"
    if [ -f "$RUN_CFG" ]; then
        echo "config exists: $RUN_CFG (reusing)"
        return
    fi
    "$PY" - <<PY
import yaml
cfg = yaml.safe_load(open('$SRC_CFG'))
cfg['output_root'] = '$OUTPUT_ROOT'
cfg['training']['batch_size'] = 4
$EXTRA
yaml.safe_dump(cfg, open('$RUN_CFG', 'w'), default_flow_style=False, sort_keys=False)
print('config:', '$RUN_CFG')
PY
}

# --------------------------------------------------------------------
# Multi-node DDP support (TSUBAME 4.0 + UGE openmpi PE).
# --------------------------------------------------------------------
#
# Usage from a multi-node job script:
#   #$ -l node_f=N -pe openmpi N
#   source "$(dirname "$0")/_common.sh"
#   tsubame_setup_env
#   tsubame_setup_multinode_env       # populates NNODES / NODE_RANK / MASTER_ADDR / NCCL env
#   mpirun -n "$NNODES" -ppn 1 -hostfile "$PE_HOSTFILE" \
#     bash -c 'tsubame_torchrun_multinode train.py --config ...'
#
# Inside the mpirun-spawned bash, OMPI_COMM_WORLD_RANK is set per node,
# so re-sourcing this and calling tsubame_setup_multinode_env gives the
# right NODE_RANK on each node.

tsubame_setup_multinode_env() {
    # NCCL env tuning for TSUBAME 4.0 (Mellanox HDR200 InfiniBand).
    # Values are safe defaults; override via env if the cluster reports
    # different HCA / interface names. NCCL auto-detects mlx5 devices.
    export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5}"
    export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ib0}"
    export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
    # Async error handling helps surface NCCL hangs early.
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    # Avoid IPv6 issues on some TSUBAME nodes.
    export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"

    if [ -n "${PE_HOSTFILE:-}" ] && [ -f "$PE_HOSTFILE" ]; then
        # UGE openmpi PE allocates N nodes; PE_HOSTFILE lists hostnames
        # (one line per node, format: "hostname N hostgroup queue").
        local NODES
        NODES=$(awk '{print $1}' "$PE_HOSTFILE")
        export NNODES=$(echo "$NODES" | wc -l)
        export MASTER_ADDR=$(echo "$NODES" | head -1)
        # mpirun -ppn 1 spawns one task per node; OMPI rank == node rank.
        export NODE_RANK="${OMPI_COMM_WORLD_RANK:-${PMIX_RANK:-0}}"
    else
        # Single-node fallback (no UGE PE allocation).
        export NNODES="${NNODES:-1}"
        export NODE_RANK="${NODE_RANK:-0}"
        export MASTER_ADDR="${MASTER_ADDR:-localhost}"
    fi
    export MASTER_PORT="${MASTER_PORT:-29500}"
    export NPROC_PER_NODE=$(nvidia-smi -L | wc -l)

    echo "[multinode] nodes=$NNODES rank=$NODE_RANK master=$MASTER_ADDR:$MASTER_PORT nproc_per_node=$NPROC_PER_NODE"
}

# Run torchrun in multi-node mode. Call after tsubame_setup_multinode_env.
# All args are passed through to the entrypoint script.
tsubame_torchrun_multinode() {
    "$TORCHRUN" --nproc_per_node="$NPROC_PER_NODE" \
                --nnodes="$NNODES" \
                --node_rank="$NODE_RANK" \
                --master_addr="$MASTER_ADDR" \
                --master_port="$MASTER_PORT" \
                "$@"
}
