#!/usr/bin/env bash
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=24:00:00
#$ -N m1cd-tsubame
#$ -j y
#$ -o /gs/fs/tga-koike-shanda2/sk/scratch/tsubame_logs/m1_cd.qsub.log
#
# TSUBAME M1 CD (Consistency Distillation) training on H100 96GB node_f (4 GPU).
#   - num_timesteps=50, single-stage, 15000 steps
#
# Submit: qsub -ar 6925 -g tga-koike-shanda m1_cd.sh

source "$(dirname "$0")/_common.sh"
tsubame_log_init "m1_cd"
tsubame_setup_env

RUN_NAME="m1_cd_tsubame_$(date +%Y%m%d_%H%M)"
OUTPUT_ROOT="$TSUBAME_ROOT/scratch/distill_methods/$RUN_NAME"
RUN_CFG="$OUTPUT_ROOT/config.yaml"

cd "$REPO/models/hunyuan3d21/hy3dshape"
tsubame_print_banner "M1 CD" "$RUN_NAME" "$OUTPUT_ROOT"
tsubame_make_run_config "$OUTPUT_ROOT" "$RUN_CFG"

NGPU=$(nvidia-smi -L | wc -l)
if [ "$NGPU" -gt 1 ]; then
    "$TORCHRUN" --nproc_per_node="$NGPU" \
        "$REPO/distill_methods/src/train.py" \
        --config "$RUN_CFG" --method cd \
        --output-dir "$OUTPUT_ROOT"
else
    "$PY" "$REPO/distill_methods/src/train.py" \
        --config "$RUN_CFG" --method cd \
        --output-dir "$OUTPUT_ROOT"
fi

echo "================================================================"
echo "M1 CD done: $(date -Iseconds)"
echo "checkpoints: $OUTPUT_ROOT/checkpoints/cd/"
echo "================================================================"
