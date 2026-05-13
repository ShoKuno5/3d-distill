#!/usr/bin/env bash
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=24:00:00
#$ -N m1dmd2-tsubame
#$ -j y
#$ -o /gs/fs/tga-koike-shanda/kuno/scratch/tsubame_logs/m1_dmd2.qsub.log
#
# TSUBAME M1 DMD2 (Distribution Matching Distillation + GAN) on H100 96GB node_f.
#   - 4 GPU DDP, effective batch=16 (per-GPU 4)
#   - 15000 steps, replay_buffer=1024
#   - DMD2 memory at batch=4 ~ 79 GB; fits in 96 GB H100.
#
# Submit: qsub -ar 6925 -g tga-koike-shanda m1_dmd2.sh

source "$(dirname "$0")/_common.sh"
tsubame_log_init "m1_dmd2"
tsubame_setup_env

RUN_NAME="m1_dmd2_tsubame_$(date +%Y%m%d_%H%M)"
OUTPUT_ROOT="$TSUBAME_ROOT/scratch/distill_methods/$RUN_NAME"
RUN_CFG="$OUTPUT_ROOT/config.yaml"

cd "$REPO/models/hunyuan3d21/hy3dshape"
tsubame_print_banner "M1 DMD2" "$RUN_NAME" "$OUTPUT_ROOT"
tsubame_make_run_config "$OUTPUT_ROOT" "$RUN_CFG"

NGPU=$(nvidia-smi -L | wc -l)
if [ "$NGPU" -gt 1 ]; then
    "$TORCHRUN" --nproc_per_node="$NGPU" \
        "$REPO/distill_methods/src/train.py" \
        --config "$RUN_CFG" --method dmd2 \
        --output-dir "$OUTPUT_ROOT"
else
    "$PY" "$REPO/distill_methods/src/train.py" \
        --config "$RUN_CFG" --method dmd2 \
        --output-dir "$OUTPUT_ROOT"
fi

echo "================================================================"
echo "M1 DMD2 done: $(date -Iseconds)"
echo "checkpoints: $OUTPUT_ROOT/checkpoints/dmd2/"
echo "================================================================"
