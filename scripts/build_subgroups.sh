#!/usr/bin/env bash
# Generate subgroup URDFs and cricket configs from the base robot description.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

python "$SCRIPT_DIR/build_subgroup_descriptions.py" \
    --input-dir "$ROOT/resources/robot/autolife" \
    --output-dir "$ROOT/resources/robot/autolife" \
    --cricket-dir "$ROOT/third_party/cricket/resources" \
    "$@"
