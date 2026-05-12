#!/usr/bin/env bash
# qzcli R1 reproduction job: DMD2 on Toys4k 420 train → Toys4k 105 eval
#
# This replicates DSW DMD2 (runs/20260414_dmd2_fix/exp1, CD=9.7e-3) on the
# qzcli H100 environment. Same model, same data, same hyperparameters as
# DSW. Diff vs DSW value isolates env effects (L20X → H100, CUDA, BF16
# numerics) before running M1 with shifted data domain.
#
# Pairs with M1 (HSSD 420 → Toys4k 105): both jobs have identical env,
# only training data domain differs, so result diff is unambiguously
# domain shift.
#
# IMPORTANT: this run intentionally uses Toys4k as training data,
# violating the normal eval-only rule. The train.py --allow-toys4k-train
# flag is required to bypass the guard.

set -euo pipefail

SK5=/inspire/qb-ilm/project/qproject-assement/zhangkaipeng-24043/sk5
REPO="$SK5/repos/3d-gen-eval"
H="$REPO/models/hunyuan3d21"
RUN_NAME="r1_dmd2_$(date +%Y%m%d_%H%M)"
OUTPUT_ROOT="$SK5/scratch/distill_methods/$RUN_NAME"

cd "$H/hy3dshape"
export HF_HOME="$SK5/hf_cache"
export PYTHONPATH=".:$REPO/distill_methods/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

[ -f "$SK5/.env" ] && set -a && . "$SK5/.env" && set +a
: "${WANDB_API_KEY:?WANDB_API_KEY must be set}"

NGPU=$(nvidia-smi -L | wc -l)
echo "================================================================"
echo "R1 DMD2 reproduction: $RUN_NAME"
echo "host: $(hostname)  ngpu: $NGPU"
echo "================================================================"

# Build R1 config from M1 config + swap train manifest + R1-specific output
CFG="$REPO/distill_methods/configs/config_525_hssd.yaml"
RUN_CFG="$OUTPUT_ROOT/config.yaml"
mkdir -p "$OUTPUT_ROOT"
"$H/.venv/bin/python" -c "
import yaml, os
cfg = yaml.safe_load(open('$CFG'))
cfg['dataset']['name'] = 'toys4k_train_420'
cfg['dataset']['manifest'] = '$REPO/distill_methods/manifests/toys4k_train_420/train.csv'
# eval is the SAME Toys4k 105 (toys4k_eval_105)
cfg['output_root'] = '$OUTPUT_ROOT'
cfg['training']['training_data_dir'] = '$OUTPUT_ROOT/training_data'
cfg['training']['methods']['dmd1']['pairs_dir'] = '$OUTPUT_ROOT/dmd1_pairs'
cfg['training']['batch_size'] = 4
yaml.safe_dump(cfg, open('$RUN_CFG', 'w'), default_flow_style=False, sort_keys=False)
print('config: $RUN_CFG')
"

# R1 needs Toys4k VAE encoding done first (different from M1's HSSD encoding)
if [ ! -d "$OUTPUT_ROOT/training_data" ] || [ "$(ls -A $OUTPUT_ROOT/training_data 2>/dev/null | wc -l)" -lt 420 ]; then
    echo "Encoding Toys4k 420 train latents..."
    CUDA_VISIBLE_DEVICES=0 "$H/.venv/bin/python" \
        "$REPO/distill_methods/scripts/prepare_training_data.py" \
        --config "$RUN_CFG"
fi

# Launch training with --allow-toys4k-train
if [ "$NGPU" -gt 1 ]; then
    echo "Launching DDP with $NGPU GPUs"
    "$H/.venv/bin/torchrun" --nproc_per_node="$NGPU" \
        "$REPO/distill_methods/src/train.py" \
        --config "$RUN_CFG" \
        --method dmd2 \
        --output-dir "$OUTPUT_ROOT" \
        --allow-toys4k-train
else
    echo "Single GPU run"
    "$H/.venv/bin/python" \
        "$REPO/distill_methods/src/train.py" \
        --config "$RUN_CFG" \
        --method dmd2 \
        --output-dir "$OUTPUT_ROOT" \
        --allow-toys4k-train
fi

echo "================================================================"
echo "R1 DMD2 done: $(date -Iseconds)"
echo "checkpoints: $OUTPUT_ROOT/checkpoints/dmd2/"
echo "================================================================"
