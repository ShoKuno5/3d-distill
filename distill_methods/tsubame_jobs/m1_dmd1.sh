#!/usr/bin/env bash
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=24:00:00
#$ -N m1dmd1-tsubame
#$ -j y
#$ -o /gs/fs/tga-koike-shanda/kuno/scratch/tsubame_logs/m1_dmd1.qsub.log
#
# TSUBAME M1 DMD1 (Distribution Matching Distillation, multistep) on H100 96GB.
#   - Step 1: pair generation if missing (skipped if pairs_dir already populated)
#   - Step 2: 15000-step training with shared pairs (4 GPU DDP)
#
# Pre-req: 20K DMD1 pairs synced into
#   $TSUBAME_ROOT/data/525_hssd/dmd1_pairs/pair_*.npz
# (Generating from scratch on TSUBAME would take ~1-2h on a single H100;
#  cheaper to rsync from sk-train.)
#
# Submit: qsub -ar 6925 -g tga-koike-shanda m1_dmd1.sh

source "$(dirname "$0")/_common.sh"
tsubame_log_init "m1_dmd1"
tsubame_setup_env

RUN_NAME="m1_dmd1_tsubame_$(date +%Y%m%d_%H%M)"
OUTPUT_ROOT="$TSUBAME_ROOT/scratch/distill_methods/$RUN_NAME"
RUN_CFG="$OUTPUT_ROOT/config.yaml"
PAIRS_DIR="$TSUBAME_ROOT/data/525_hssd/dmd1_pairs"

cd "$REPO/models/hunyuan3d21/hy3dshape"
tsubame_print_banner "M1 DMD1" "$RUN_NAME" "$OUTPUT_ROOT"
mkdir -p "$PAIRS_DIR"
tsubame_make_run_config "$OUTPUT_ROOT" "$RUN_CFG" \
    "cfg['training']['methods']['dmd1']['pairs_dir'] = '$PAIRS_DIR'"

NUM_PAIRS=$("$PY" -c "import yaml; c=yaml.safe_load(open('$RUN_CFG')); print(c['training']['methods']['dmd1']['num_pairs'])")
HAVE_PAIRS=$(ls "$PAIRS_DIR"/pair_*.npz 2>/dev/null | wc -l)
echo "DMD1 pairs: have=$HAVE_PAIRS  need=$NUM_PAIRS"

if [ "$HAVE_PAIRS" -lt "$NUM_PAIRS" ]; then
    echo "Generating DMD1 pairs (single shard, may take 1-2h on 1 H100)..."
    CUDA_VISIBLE_DEVICES=0 "$PY" \
        "$REPO/distill_methods/scripts/generate_dmd1_pairs.py" \
        --config "$RUN_CFG" --shard 0 --num-shards 1
    HAVE_PAIRS=$(ls "$PAIRS_DIR"/pair_*.npz 2>/dev/null | wc -l)
    echo "After pair gen: have=$HAVE_PAIRS"
fi

NGPU=$(nvidia-smi -L | wc -l)
echo "================================================================"
echo "DMD1 training start: $(date -Iseconds)  ngpu=$NGPU"
echo "================================================================"
if [ "$NGPU" -gt 1 ]; then
    "$TORCHRUN" --nproc_per_node="$NGPU" \
        "$REPO/distill_methods/src/train.py" \
        --config "$RUN_CFG" --method dmd1 \
        --output-dir "$OUTPUT_ROOT"
else
    "$PY" "$REPO/distill_methods/src/train.py" \
        --config "$RUN_CFG" --method dmd1 \
        --output-dir "$OUTPUT_ROOT"
fi

echo "================================================================"
echo "M1 DMD1 done: $(date -Iseconds)"
echo "checkpoints: $OUTPUT_ROOT/checkpoints/dmd1/"
echo "================================================================"
