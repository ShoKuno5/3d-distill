#!/usr/bin/env bash
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=2:00:00
#$ -N teacher-inference
#$ -j y
#$ -o /gs/fs/tga-koike-shanda2/sk/scratch/tsubame_logs/eval_teacher_inference.qsub.log
#
# teacher_50step (HY3D-2.1 50-step) inference on Toys4k 105 sample.
# Uses existing 3d-distill venv + HY3D-2.1 HF cache (already loaded for M2 CD smoke).
# Output: predictions/teacher_50step/default/<oid>/mesh_raw.obj

source "${COMMON_SH:-/gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/tsubame_jobs/_common.sh}"
tsubame_log_init "eval_teacher_inference"
tsubame_setup_env

REPO_MAIN=/gs/fs/tga-koike-shanda2/sk/3d-distill
REPO_EVAL=/gs/fs/tga-koike-shanda2/sk/3d-distill-eval
CFG=$REPO_EVAL/distill_methods/config_tsubame_cd_eval.yaml

echo "================================================================"
echo "teacher_50step inference: $(date -Iseconds), host=$(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader -i 0
echo "config: $CFG"
echo "================================================================"

cd "$REPO_MAIN/models/hunyuan3d21/hy3dshape"
export PYTHONPATH="$REPO_MAIN/models/hunyuan3d21/hy3dshape:$REPO_EVAL/pipeline:$REPO_EVAL/distill_methods:$PYTHONPATH"

CUDA_VISIBLE_DEVICES=0 "$PY" \
    "$REPO_EVAL/distill_methods/scripts/run_inference_distilled.py" \
    --config "$CFG" \
    --model-name teacher_50step

echo "================================================================"
echo "teacher_50step inference done: $(date -Iseconds)"
M2=/gs/fs/tga-koike-shanda2/sk/scratch/distill_methods/m2_cd_tsubame_20260519_1629
echo "predictions:"
find $M2/predictions/teacher_50step -name "mesh_raw.obj" 2>/dev/null | wc -l
echo "================================================================"
