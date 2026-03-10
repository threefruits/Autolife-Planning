#!/usr/bin/env bash
# Generate FK/collision-checking C++ code for vamp using cricket.
# All project-specific paths live here.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

CRICKET="$ROOT/third_party/cricket"
VAMP="$ROOT/third_party/vamp"
RESOURCES="$ROOT/resources/robot/autolife"

# All robot variants: the original full-body + subgroups
ROBOTS=(autolife)
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
)

# 1. Distribute robot descriptions to cricket + vamp resources
echo "[1/3] Distributing robot descriptions..."
for DEST in "$CRICKET/resources/autolife" "$VAMP/resources/autolife"; do
    cp "$RESOURCES/autolife_spherized.urdf" "$DEST/autolife_spherized.urdf"
    cp "$RESOURCES/autolife.srdf" "$DEST/autolife.srdf"
    for name in "${SUBGROUPS[@]}"; do
        SFILE="$RESOURCES/${name}_spherized.urdf"
        if [ -f "$SFILE" ]; then
            cp "$SFILE" "$DEST/${name}_spherized.urdf"
        fi
    done
    echo "  copied to $DEST"
done

# 2. Run cricket FK code generation for all variants
echo "[2/3] Generating FK code..."
"$CRICKET/build/fkcc_gen" "$CRICKET/resources/autolife.json"
for name in "${SUBGROUPS[@]}"; do
    CONFIG="$CRICKET/resources/${name}.json"
    if [ -f "$CONFIG" ]; then
        echo "  generating FK for $name..."
        "$CRICKET/build/fkcc_gen" "$CONFIG"
    fi
done

# 3. Copy generated headers to vamp
echo "[3/3] Installing FK headers into vamp..."
cp "autolife_fk.hh" "$VAMP/src/impl/vamp/robots/autolife.hh"
echo "  installed autolife.hh"
for name in "${SUBGROUPS[@]}"; do
    HEADER="${name}_fk.hh"
    if [ -f "$HEADER" ]; then
        cp "$HEADER" "$VAMP/src/impl/vamp/robots/${name}.hh"
        echo "  installed ${name}.hh"
    fi
done

echo "Done! Rebuild vamp to use the new FK code."
