#!/usr/bin/env bash
# resolution_sweep: 入力解像度(300/512/1024)の影響を調査
#
# Usage:
#   bash experiments/resolution_sweep/run.sh                  # 全解像度をフル実行
#   bash experiments/resolution_sweep/run.sh --skip-inference  # 評価のみ
#   bash experiments/resolution_sweep/run.sh --only=512        # 特定解像度のみ
#   bash experiments/resolution_sweep/run.sh --render-grid     # 比較グリッド画像生成

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Parse --only and --render-grid, pass rest through
ONLY=""
RENDER_GRID=false
PASSTHROUGH=()
for arg in "$@"; do
    case $arg in
        --only=*) ONLY="${arg#*=}" ;;
        --render-grid) RENDER_GRID=true ;;
        *) PASSTHROUGH+=("$arg") ;;
    esac
done

RESOLUTIONS=(300 512 1024)
if [ -n "$ONLY" ]; then
    RESOLUTIONS=("$ONLY")
fi

for res in "${RESOLUTIONS[@]}"; do
    config="$SCRIPT_DIR/res_${res}.yaml"
    if [ ! -f "$config" ]; then
        echo "ERROR: Config not found: $config"
        continue
    fi
    echo ""
    echo "========================================="
    echo "Resolution: ${res}×${res}"
    echo "========================================="
    bash "$PROJECT_DIR/pipeline/scripts/run_all_eval.sh" \
        --config "$config" \
        "${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}"
done

if [ "$RENDER_GRID" = true ]; then
    echo ""
    echo "========================================="
    echo "Rendering comparison grid"
    echo "========================================="
    "$PROJECT_DIR/envs/hunyuan3d-venv/bin/python" \
        "$PROJECT_DIR/pipeline/scripts/render_grid_blender.py" --workers 16
fi
