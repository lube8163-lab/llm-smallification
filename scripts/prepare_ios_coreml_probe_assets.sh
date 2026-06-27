#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="${1:-"$ROOT_DIR/runpod-artifacts/compiled"}"
DST_DIR="$ROOT_DIR/ios/CoreMLProbe/CoreMLProbe/Models"

MODELS=(
  "gemma4_12b_embedding_seq4_int4_block32.mlmodelc"
  "gemma4_12b_layer0_decoder_seq4_mask_int4_block32.mlmodelc"
  "gemma4_12b_lm_head_1tok_int4_block32.mlmodelc"
)

mkdir -p "$DST_DIR"

for model in "${MODELS[@]}"; do
  if [[ ! -d "$SRC_DIR/$model" ]]; then
    echo "missing: $SRC_DIR/$model" >&2
    exit 1
  fi
done

for model in "${MODELS[@]}"; do
  rm -rf "$DST_DIR/$model"
  ditto --norsrc --noextattr "$SRC_DIR/$model" "$DST_DIR/$model"
  xattr -cr "$DST_DIR/$model" 2>/dev/null || true
  du -sh "$DST_DIR/$model"
done

xattr -cr "$DST_DIR" 2>/dev/null || true

echo "copied ${#MODELS[@]} model bundles into $DST_DIR"
