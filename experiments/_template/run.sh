#!/usr/bin/env bash
# <experiment_name>: <one-line description>
#
# Usage:
#   bash experiments/<experiment_name>/run.sh                  # full run
#   bash experiments/<experiment_name>/run.sh --skip-inference  # eval only
#   bash experiments/<experiment_name>/run.sh --workers=8       # set parallel workers

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

exec bash "$PROJECT_DIR/pipeline/scripts/run_all_eval.sh" \
    --config "$SCRIPT_DIR/config.yaml" \
    "$@"
