#!/usr/bin/env bash
# toys4k_baseline: 4モデルベースライン比較（10サンプル）
#
# Usage:
#   bash experiments/toys4k_baseline/run.sh                  # フル実行
#   bash experiments/toys4k_baseline/run.sh --skip-inference  # 評価のみ
#   bash experiments/toys4k_baseline/run.sh --workers=8       # 並列ワーカー数指定

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

exec bash "$PROJECT_DIR/pipeline/scripts/run_all_eval.sh" \
    --config "$SCRIPT_DIR/config.yaml" \
    "$@"
