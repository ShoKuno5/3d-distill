#!/usr/bin/env bash
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=0:30:00
#$ -N m2-smoke-batch
#$ -j y
#$ -o /gs/fs/tga-koike-shanda2/sk/scratch/tsubame_logs/m2_smoke_batch.qsub.log
#
# M2 DMD2 batch headroom smoke: 1 GPU, batch=4, 20 steps, peak VRAM probe.

source "${COMMON_SH:-/gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/tsubame_jobs/_common.sh}"
tsubame_log_init "m2_smoke_batch"
tsubame_setup_env

RUN_NAME="m2_smoke_batch_$(date +%Y%m%d_%H%M)"
OUTPUT_ROOT="$TSUBAME_ROOT/scratch/distill_methods/$RUN_NAME"
RUN_CFG="$OUTPUT_ROOT/config.yaml"
GPUMEM_LOG="$OUTPUT_ROOT/gpumem.csv"
mkdir -p "$OUTPUT_ROOT"

# Build smoke config from M2 base with very small total_steps.
"$PY" - <<PYEOF
import yaml, pathlib
src = "$REPO/distill_methods/configs/config_5k_m2_tsubame.yaml"
with open(src) as f: c = yaml.safe_load(f)
c["training"]["total_steps"] = 20
c["training"]["save_interval"] = 9999
c["training"]["log_interval"] = 2
c["output_root"] = "$OUTPUT_ROOT"
with open("$RUN_CFG", "w") as f: yaml.safe_dump(c, f, sort_keys=False)
print(f"smoke config written: $RUN_CFG")
print("  batch_size =", c["training"]["batch_size"])
print("  total_steps =", c["training"]["total_steps"])
PYEOF

echo "================================================================"
echo "M2 DMD2 smoke (batch=4, 1 GPU): $(date -Iseconds)"
echo "host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
echo "================================================================"

# Background GPU memory poller (every 1s).
nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu --format=csv -l 1 > "$GPUMEM_LOG" &
NVSMI_PID=$!
trap "kill $NVSMI_PID 2>/dev/null || true" EXIT

cd "$REPO/models/hunyuan3d21/hy3dshape"
export CUDA_VISIBLE_DEVICES=0
"$TORCHRUN" --standalone --nproc_per_node=1 \
    "$REPO/distill_methods/src/train.py" \
    --config "$RUN_CFG" --method dmd2 \
    --output-dir "$OUTPUT_ROOT"

TRAIN_EXIT=$?
kill $NVSMI_PID 2>/dev/null || true
sleep 1

echo "================================================================"
echo "Train exit: $TRAIN_EXIT"
echo "=== peak GPU memory ==="
awk -F"," "NR>1 && \$2 ~ /MiB/ { gsub(/ MiB/, \"\", \$2); v=\$2+0; if(v>max) max=v } END { printf \"peak used = %d MiB (%.1f GB) of 96GB\n\", max, max/1024 }" "$GPUMEM_LOG"
tail -5 "$GPUMEM_LOG"
echo "================================================================"
