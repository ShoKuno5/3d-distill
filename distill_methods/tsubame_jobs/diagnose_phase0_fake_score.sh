#!/usr/bin/env bash
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=1:30:00
#$ -N dmd2-diag-fakescore
#$ -j y
#$ -o /gs/fs/tga-koike-shanda2/sk/scratch/tsubame_logs/diagnose_phase0_fake_score.qsub.log
#
# Phase 0a/b: fake_score calibration (C1) + same-vs-fresh batch overfit (C3).
# Reuses 1 GPU on node_f (other 3 idle — node_f is the smallest node
# allocation TSUBAME exposes for the hy3dshape stack).
#
#   qsub -g tga-koike-shanda diagnose_phase0_fake_score.sh
#
# Outputs in /gs/fs/tga-koike-shanda2/sk/scratch/phase0a_*.{txt,png} and
# /gs/fs/tga-koike-shanda2/sk/scratch/phase0b_overfit.csv.

source "${COMMON_SH:-/gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/tsubame_jobs/_common.sh}"
tsubame_log_init "diagnose_phase0_fake_score"
tsubame_setup_env
export HY3DGEN_MODELS=/gs/fs/tga-koike-shanda2/sk/cache/hy3dgen

REPO_MAIN=/gs/fs/tga-koike-shanda2/sk/3d-distill
CFG=$REPO_MAIN/distill_methods/configs/config_5k_m2_tsubame.yaml
CKPT=/gs/fs/tga-koike-shanda2/sk/scratch/distill_methods/m2_dmd2_multinode_20260519_1022/checkpoints/dmd2/step_12000

cd "$REPO_MAIN/models/hunyuan3d21/hy3dshape"
export PYTHONPATH="$REPO_MAIN/models/hunyuan3d21/hy3dshape:$REPO_MAIN/distill_methods/src:$PYTHONPATH"

echo "================================================================"
echo "Phase 0a/b fake_score diagnosis: $(date -Iseconds), host=$(hostname)"
nvidia-smi -L
echo "config: $CFG"
echo "ckpt:   $CKPT"
echo "================================================================"

if [ ! -d "$CKPT" ]; then
    echo "ERROR: step_12000 ckpt missing: $CKPT" >&2
    exit 1
fi

CUDA_VISIBLE_DEVICES=0 "$PY" \
    /gs/fs/tga-koike-shanda2/sk/scratch/diagnose_fake_score.py \
    --config "$CFG" \
    --student-ckpt "$CKPT" \
    --warmup-steps 100 \
    --measure-batches 10 \
    --out-dir /gs/fs/tga-koike-shanda2/sk/scratch

RC=$?
echo "diagnose_fake_score.py exit: $RC"
echo "================================================================"
echo "Phase 0a/b done: $(date -Iseconds)"
echo "================================================================"
exit $RC
