#!/usr/bin/env bash
# Spherize all subgroup URDFs using foam.
# Each subgroup URDF contains the full robot body (all links, all collision
# geometry).  Only the joint types differ: non-planning joints are fixed.
# Spherization produces sphere approximations for EVERY link so that the
# planner checks collisions against the entire body.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

RESOURCES="$ROOT/resources/robot/autolife"
FOAM="$ROOT/third_party/foam"

SUBGROUPS=(
    autolife_left_high
    autolife_left_mid
    autolife_left_low
    autolife_right_high
    autolife_right_mid
    autolife_right_low
    autolife_dual_high
    autolife_dual_mid
    autolife_dual_low
    autolife_torso_left_high
    autolife_torso_left_mid
    autolife_torso_left_low
    autolife_torso_right_high
    autolife_torso_right_mid
    autolife_torso_right_low
    autolife_body
    autolife_body_coupled
)

for name in "${SUBGROUPS[@]}"; do
    INPUT="$RESOURCES/${name}.urdf"
    OUTPUT="$RESOURCES/${name}_spherized.urdf"
    if [ ! -f "$INPUT" ]; then
        echo "SKIP $name (${INPUT} not found)"
        continue
    fi
    echo "Spherizing $name..."
    python "$FOAM/scripts/generate_sphere_urdf.py" \
        "$INPUT" \
        --output "$OUTPUT" \
        --database "$FOAM/sphere_database.json" \
        "$@"
done

echo "Done spherizing subgroups."
