#!/usr/bin/env bash
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=0:30:00
#$ -N render-blank-test
#$ -j y
#$ -o /gs/fs/tga-koike-shanda2/sk/scratch/tsubame_logs/render_blank_test.qsub.log

source /gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/tsubame_jobs/_common.sh
tsubame_log_init render_blank_test
tsubame_setup_env

export BLENDER_BIN="$TSUBAME_ROOT/blender-3.6.18-linux-x64/blender"
MANIFEST="$REPO/distill_methods/manifests/5k_balanced/blank_test_1.csv"
OBJ_DIR="$TSUBAME_ROOT/data/5k_balanced/objaverse_sketchfab/e4bb4fdd830d460b9f70ef6f43de4706"

echo "=== before ==="
ls -la "$OBJ_DIR/image.png" 2>/dev/null
md5sum "$OBJ_DIR/image.png" 2>/dev/null
[ -f "$OBJ_DIR/image.png" ] && mv "$OBJ_DIR/image.png" "$OBJ_DIR/image.png.blank_bak"

echo "=== rendering with timeout=600s ==="
CUDA_VISIBLE_DEVICES=0 "$PY" "$REPO/distill_methods/scripts/render_manifest.py" \
    --manifest "$MANIFEST" \
    --resolution 512 \
    --timeout-sec 600 \
    --shard 0 --num-shards 1

echo "=== after ==="
ls -la "$OBJ_DIR/image.png" 2>/dev/null
md5sum "$OBJ_DIR/image.png" 2>/dev/null
