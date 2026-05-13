#!/usr/bin/env bash
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=24:00:00
#$ -N m1pd-tsubame
#$ -j y
#$ -o /gs/fs/tga-koike-shanda/kuno/scratch/tsubame_logs/m1_pd.qsub.log
#
# TSUBAME M1 PD (Progressive Distillation) training on H100 96GB node_f (4 GPU).
#   - 3 stages [50->25, 25->12, 12->6], steps_per_stage=5000
#   - Idempotent: stages with merged ckpt are skipped on re-run.
#
# Submit: qsub -ar 6925 -g tga-koike-shanda m1_pd.sh

source "$(dirname "$0")/_common.sh"
tsubame_log_init "m1_pd"
tsubame_setup_env

RUN_NAME="${M1_PD_RUN_NAME:-m1_pd_tsubame_$(date +%Y%m%d_%H%M)}"
OUTPUT_ROOT="${M1_PD_OUTPUT_ROOT:-$TSUBAME_ROOT/scratch/distill_methods/$RUN_NAME}"
RUN_CFG="$OUTPUT_ROOT/config.yaml"

cd "$REPO/models/hunyuan3d21/hy3dshape"
tsubame_print_banner "M1 PD" "$RUN_NAME" "$OUTPUT_ROOT"
tsubame_make_run_config "$OUTPUT_ROOT" "$RUN_CFG"

NSTAGES=$("$PY" -c "import yaml; print(len(yaml.safe_load(open('$RUN_CFG'))['training']['methods']['pd']['stages']))")
NGPU=$(nvidia-smi -L | wc -l)
echo "PD: $NSTAGES stages total, ngpu=$NGPU"

for STAGE in $(seq 0 $((NSTAGES - 1))); do
    MERGED="$OUTPUT_ROOT/checkpoints/pd/stage_${STAGE}_merged/model.pt"
    if [ -f "$MERGED" ]; then
        echo "[stage $STAGE] already complete (merged ckpt exists), skipping"
        continue
    fi
    echo "================================================================"
    echo "PD stage $STAGE start: $(date -Iseconds)"
    echo "================================================================"
    if [ "$NGPU" -gt 1 ]; then
        "$TORCHRUN" --nproc_per_node="$NGPU" \
            "$REPO/distill_methods/src/train.py" \
            --config "$RUN_CFG" --method pd --stage "$STAGE" \
            --output-dir "$OUTPUT_ROOT"
    else
        "$PY" "$REPO/distill_methods/src/train.py" \
            --config "$RUN_CFG" --method pd --stage "$STAGE" \
            --output-dir "$OUTPUT_ROOT"
    fi
done

echo "================================================================"
echo "M1 PD done: $(date -Iseconds)"
echo "checkpoints: $OUTPUT_ROOT/checkpoints/pd/"
echo "================================================================"
