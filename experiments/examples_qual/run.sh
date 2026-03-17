#!/usr/bin/env bash
# examples_qual: 定性的な出力例の生成
#
# Usage:
#   bash experiments/examples_qual/run.sh                  # フル実行
#   bash experiments/examples_qual/run.sh --skip-inference  # 評価のみ

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

exec bash "$PROJECT_DIR/pipeline/scripts/run_all_eval.sh" \
    --config "$SCRIPT_DIR/config.yaml" \
    "$@"
