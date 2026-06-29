#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENDPOINTS_ONLY=0
if [[ "${1:-}" == "--endpoints-only" ]]; then
  ENDPOINTS_ONLY=1
  shift
fi
SRC_DIR="${1:-"$ROOT_DIR/runpod-artifacts/compiled"}"
DST_DIR="${DST_DIR:-"$ROOT_DIR/ios/CoreMLProbe/CoreMLProbe/Models"}"
SEQ_LEN="${SEQ_LEN:-4}"

EMBEDDING_MODEL="gemma4_12b_embedding_seq${SEQ_LEN}_int4_block32.mlmodelc"
NORM_LM_HEAD_MODEL="gemma4_12b_norm_lm_head_1tok_int4_block32.mlmodelc"
LEGACY_LM_HEAD_MODEL="gemma4_12b_lm_head_1tok_int4_block32.mlmodelc"
MODELS=()

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

if [[ "$ENDPOINTS_ONLY" != "1" || -d "$SRC_DIR/$EMBEDDING_MODEL" ]]; then
  MODELS+=("$EMBEDDING_MODEL")
fi

DECODER_MODELS=()
if [[ "$ENDPOINTS_ONLY" != "1" ]]; then
  while IFS= read -r model; do
    DECODER_MODELS+=("$model")
  done < <(
    {
      find "$SRC_DIR" -maxdepth 1 -type d \
        -name "gemma4_12b_layer[0-9][0-9]_decoder_seq${SEQ_LEN}_mask_int4_block32.mlmodelc" \
        -exec basename {} \;
      find "$SRC_DIR" -maxdepth 1 -type d \
        -name "gemma4_12b_layers[0-9][0-9]_[0-9][0-9]_decoder_seq${SEQ_LEN}_mask_int4_block32.mlmodelc" \
        -exec basename {} \;
    } | sort
  )

  if [[ "${#DECODER_MODELS[@]}" -eq 0 ]]; then
    echo "missing decoder layers/chunks: $SRC_DIR/gemma4_12b_layer*_decoder_seq${SEQ_LEN}_mask_int4_block32.mlmodelc" >&2
    exit 1
  fi
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

if [[ "$ENDPOINTS_ONLY" != "1" ]]; then
  find "$DST_DIR" -maxdepth 1 -type d \
    -name "gemma4_12b_layer[0-9][0-9]_decoder_seq${SEQ_LEN}_mask_int4_block32.mlmodelc" \
    -exec rm -rf {} +
  find "$DST_DIR" -maxdepth 1 -type d \
    -name "gemma4_12b_layers[0-9][0-9]_[0-9][0-9]_decoder_seq${SEQ_LEN}_mask_int4_block32.mlmodelc" \
    -exec rm -rf {} +
fi

COPY_MODELS=()
for model in "${MODELS[@]}"; do
  COPY_MODELS+=("$model")
done
if [[ "${#DECODER_MODELS[@]}" -gt 0 ]]; then
  for model in "${DECODER_MODELS[@]}"; do
    COPY_MODELS+=("$model")
  done
fi

for model in "${COPY_MODELS[@]}"; do
  rm -rf "$DST_DIR/$model"
  ditto --norsrc --noextattr "$SRC_DIR/$model" "$DST_DIR/$model"
  clear_packaging_xattrs "$DST_DIR/$model"
  du -sh "$DST_DIR/$model"
done

clear_packaging_xattrs "$DST_DIR"

if [[ "$ENDPOINTS_ONLY" == "1" ]]; then
  echo "updated ${#COPY_MODELS[@]} endpoint model bundles in $DST_DIR"
else
  echo "copied ${#COPY_MODELS[@]} model bundles into $DST_DIR"
fi
