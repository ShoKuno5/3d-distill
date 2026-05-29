#!/usr/bin/env bash
#$ -cwd
#$ -l gpu_1
#$ -l h_rt=2:00:00
#$ -N m2cd-eval-inference
#$ -j y
#$ -o /gs/fs/tga-koike-shanda2/sk/scratch/tsubame_logs/m2_cd_eval_inference.qsub.log
#
# M2 CD eval: inference on Toys4k 105 test set using M2 CD ckpt.
#   - Uses cross-family-bench worktree: scripts + pipeline utils
#   - LoRA from /gs/fs/.../m2_cd_tsubame_20260519_1629/checkpoints/cd/step_final
#   - 1 GPU (gpu_1), h_rt 2h (~10-20 min expected)

source "${COMMON_SH:-/gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/tsubame_jobs/_common.sh}"
tsubame_log_init "m2_cd_eval_inference"
tsubame_setup_env

REPO_MAIN=/gs/fs/tga-koike-shanda2/sk/3d-distill
REPO_EVAL=/gs/fs/tga-koike-shanda2/sk/3d-distill-eval
CFG=$REPO_EVAL/distill_methods/config_tsubame_cd_eval.yaml

echo "================================================================"
echo "M2 CD eval inference: $(date -Iseconds)"
echo "host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader -i 0
echo "config: $CFG"
echo "================================================================"

# hy3dshape needs to be on PYTHONPATH; cd into models/hunyuan3d21 (per upstream README convention).
cd "$REPO_MAIN/models/hunyuan3d21/hy3dshape"
export PYTHONPATH="$REPO_MAIN/models/hunyuan3d21/hy3dshape:$REPO_EVAL/pipeline:$REPO_EVAL/distill_methods:$PYTHONPATH"

CUDA_VISIBLE_DEVICES=0 "$PY" \
    "$REPO_EVAL/distill_methods/scripts/run_inference_distilled.py" \
    --config "$CFG" \
    --model-name cd_4step

echo "================================================================"
echo "M2 CD eval inference done: $(date -Iseconds)"
echo "predictions: /gs/fs/tga-koike-shanda2/sk/scratch/distill_methods/m2_cd_tsubame_20260519_1629/predictions/cd_4step/default/"
echo "================================================================"
