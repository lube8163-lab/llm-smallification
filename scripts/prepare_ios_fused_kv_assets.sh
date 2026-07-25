#!/usr/bin/env bash
# Compile/copy the measured-safe six-layer KV fusion and stage it in the iOS
# Models directory. Only layers 00...05 are installed intentionally:
# layers 06...11 made the same 24-token iPhone 17 run 5.4% slower and raised
# peak memory, so additional groups remain conversion experiments.
#
# Usage:
#   ./scripts/prepare_ios_fused_kv_assets.sh <source-directory>
#
# The source directory must contain the prefill and verify assets below as
# either .mlpackage or .mlmodelc directories. Override MODELS_DIR or
# COMPILED_DIR when staging into a temporary app/build.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DIR="${1:?source directory containing fused KV assets}"
MODELS_DIR="${MODELS_DIR:-$ROOT_DIR/ios/CoreMLProbe/CoreMLProbe/Models}"
COMPILED_DIR="${COMPILED_DIR:-$ROOT_DIR/runpod-artifacts/kv-fused6/compiled}"

MODEL_NAMES=(
  "gemma4_12b_layers00_05_prefill_seq320_kv_pal4_g16"
  "gemma4_12b_layers00_05_verify_seq4_kv512_pal4_g16"
)

mkdir -p "$MODELS_DIR" "$COMPILED_DIR"

for name in "${MODEL_NAMES[@]}"; do
  package="$SOURCE_DIR/$name.mlpackage"
  compiled_source="$SOURCE_DIR/$name.mlmodelc"
  compiled="$COMPILED_DIR/$name.mlmodelc"

  if [[ -d "$package" ]]; then
    echo "compile $name"
    rm -rf "$compiled"
    xcrun coremlcompiler compile "$package" "$COMPILED_DIR" \
      --platform iOS \
      --deployment-target 18.0 >/dev/null
  elif [[ -d "$compiled_source" ]]; then
    compiled="$compiled_source"
  else
    echo "missing $name.mlpackage or $name.mlmodelc in $SOURCE_DIR" >&2
    exit 1
  fi

  destination="$MODELS_DIR/$name.mlmodelc"
  echo "stage $name"
  rm -rf "$destination"
  ditto --norsrc --noextattr "$compiled" "$destination"
  xattr -cr "$destination" 2>/dev/null || true

  [[ -f "$destination/coremldata.bin" ]] || {
    echo "compiled asset is incomplete: $destination/coremldata.bin" >&2
    exit 1
  }
done

echo "staged measured six-layer fusion in $MODELS_DIR"
du -sh "$MODELS_DIR"/gemma4_12b_layers00_05_*.mlmodelc
