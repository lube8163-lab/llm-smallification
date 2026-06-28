#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="${1:-"$ROOT_DIR/runpod-artifacts/compiled"}"
DST_DIR="$ROOT_DIR/ios/CoreMLProbe/CoreMLProbe/Models"

EMBEDDING_MODEL="gemma4_12b_embedding_seq4_int4_block32.mlmodelc"
NORM_LM_HEAD_MODEL="gemma4_12b_norm_lm_head_1tok_int4_block32.mlmodelc"
LEGACY_LM_HEAD_MODEL="gemma4_12b_lm_head_1tok_int4_block32.mlmodelc"
MODELS=("$EMBEDDING_MODEL")

clear_packaging_xattrs() {
  local target="$1"
  [[ -e "$target" ]] || return 0
  xattr -cr "$target" 2>/dev/null || true
  find "$target" -mindepth 0 -exec sh -c '
    for path do
      xattr -d com.apple.FinderInfo "$path" 2>/dev/null || true
      xattr -d "com.apple.fileprovider.fpfs#P" "$path" 2>/dev/null || true
    done
  ' sh {} +
}

mkdir -p "$DST_DIR"

DECODER_MODELS=()
while IFS= read -r model; do
  DECODER_MODELS+=("$model")
done < <(
  find "$SRC_DIR" -maxdepth 1 -type d \
    -name "gemma4_12b_layer*_decoder_seq4_mask_int4_block32.mlmodelc" \
    -exec basename {} \; | sort
)

if [[ "${#DECODER_MODELS[@]}" -eq 0 ]]; then
  echo "missing decoder layers: $SRC_DIR/gemma4_12b_layer*_decoder_seq4_mask_int4_block32.mlmodelc" >&2
  exit 1
fi

if [[ -d "$SRC_DIR/$NORM_LM_HEAD_MODEL" ]]; then
  MODELS+=("$NORM_LM_HEAD_MODEL")
elif [[ -d "$SRC_DIR/$LEGACY_LM_HEAD_MODEL" ]]; then
  echo "warning: using legacy LM head without final norm/softcap: $LEGACY_LM_HEAD_MODEL" >&2
  MODELS+=("$LEGACY_LM_HEAD_MODEL")
else
  echo "missing: $SRC_DIR/$NORM_LM_HEAD_MODEL or $SRC_DIR/$LEGACY_LM_HEAD_MODEL" >&2
  exit 1
fi

for model in "${MODELS[@]}"; do
  if [[ ! -d "$SRC_DIR/$model" ]]; then
    echo "missing: $SRC_DIR/$model" >&2
    exit 1
  fi
done

rm -rf "$DST_DIR/$NORM_LM_HEAD_MODEL" "$DST_DIR/$LEGACY_LM_HEAD_MODEL"

find "$DST_DIR" -maxdepth 1 -type d \
  -name "gemma4_12b_layer*_decoder_seq4_mask_int4_block32.mlmodelc" \
  -exec rm -rf {} +

for model in "${MODELS[@]}" "${DECODER_MODELS[@]}"; do
  rm -rf "$DST_DIR/$model"
  ditto --norsrc --noextattr "$SRC_DIR/$model" "$DST_DIR/$model"
  clear_packaging_xattrs "$DST_DIR/$model"
  du -sh "$DST_DIR/$model"
done

clear_packaging_xattrs "$DST_DIR"

echo "copied $((${#MODELS[@]} + ${#DECODER_MODELS[@]})) model bundles into $DST_DIR"
