#!/usr/bin/env bash
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=4:00:00
#$ -N render-525-hssd
#$ -j y
#$ -o /gs/fs/tga-koike-shanda2/sk/scratch/tsubame_logs/render_525.qsub.log
#
# Render 525 HSSD manifest images via Blender 3.6 (GPU CYCLES + OPTIX).
# Each render ~12s after first-time kernel cache; ~1.75h for 525 sequential.
# Survives disconnects: scheduled as qsub batch on 1 H100 node (node_f).
#
# Submit:  qsub -ar 6925 -g tga-koike-shanda render_525_hssd.sh
# Without reservation: qsub -g tga-koike-shanda render_525_hssd.sh

source "${COMMON_SH:-/gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/tsubame_jobs/_common.sh}"
tsubame_log_init "render_525"
tsubame_setup_env

export BLENDER_BIN="$TSUBAME_ROOT/blender-3.6.18-linux-x64/blender"

cd "$REPO"
echo "================================================================"
echo "Render 525 HSSD: $(date -Iseconds)"
echo "host: $(hostname)  gpu count: $(nvidia-smi -L | wc -l)"
echo "BLENDER_BIN=$BLENDER_BIN"
echo "================================================================"

python distill_methods/scripts/render_manifest.py \
    --manifest distill_methods/manifests/525_hssd/train.csv distill_methods/manifests/525_hssd/test.csv \
    --resolution 512 \
    --timeout-sec 300

echo "================================================================"
echo "render done: $(date -Iseconds)"
echo "rendered count: $(find $TSUBAME_ROOT/data/trellis500k/hssd -name image.png 2>/dev/null | wc -l)"
echo "================================================================"
