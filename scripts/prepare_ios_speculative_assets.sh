#!/usr/bin/env bash
# Compile/verify the Gemma 4 MTP drafter and seq-K target verifier bundles,
# then stage them into the CoreMLProbe app.
#
# Usage:
#   ./scripts/prepare_ios_speculative_assets.sh <verify-dir> <drafter.mlpackage|drafter.mlmodelc>
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERIFY_SOURCE="${1:?verify bundle directory}"
DRAFTER_SOURCE="${2:?drafter .mlpackage or .mlmodelc}"
VERIFY_SEQ="${VERIFY_SEQ:-4}"
CACHE_SIZE="${CACHE_SIZE:-512}"
EXPECTED_LAYERS="${EXPECTED_LAYERS:-48}"
COMPILED_DIR="${COMPILED_DIR:-$ROOT_DIR/runpod-artifacts/speculative/compiled}"
MODELS_DIR="${MODELS_DIR:-$ROOT_DIR/ios/CoreMLProbe/CoreMLProbe/Models}"

if [[ ! -d "$VERIFY_SOURCE" ]]; then
  echo "missing verify directory: $VERIFY_SOURCE" >&2
  exit 1
fi
if [[ ! -d "$DRAFTER_SOURCE" ]]; then
  echo "missing drafter bundle: $DRAFTER_SOURCE" >&2
  exit 1
fi
DRAFTER_NAME="$(basename "$DRAFTER_SOURCE")"
DRAFTER_NAME="${DRAFTER_NAME%.mlpackage}"
DRAFTER_NAME="${DRAFTER_NAME%.mlmodelc}"
if [[ "$DRAFTER_NAME" != "gemma4_12b_drafter_step_kv${CACHE_SIZE}_mixed_pal4_g16_head_int8" ]]; then
  echo "refusing unvalidated drafter: $DRAFTER_NAME" >&2
  exit 1
fi

mkdir -p "$COMPILED_DIR" "$MODELS_DIR"

compile_or_copy() {
  local source="$1"
  local basename_no_ext
  local destination
  case "$source" in
    *.mlpackage)
      basename_no_ext="$(basename "$source" .mlpackage)"
      destination="$COMPILED_DIR/$basename_no_ext.mlmodelc"
      rm -rf "$destination"
      echo "compile: $(basename "$source")"
      xcrun coremlcompiler compile "$source" "$COMPILED_DIR" >/dev/null
      ;;
    *.mlmodelc)
      destination="$COMPILED_DIR/$(basename "$source")"
      rm -rf "$destination"
      echo "copy compiled: $(basename "$source")"
      ditto --norsrc --noextattr "$source" "$destination"
      ;;
    *)
      echo "unsupported Core ML bundle: $source" >&2
      exit 1
      ;;
  esac
  xattr -cr "$destination" 2>/dev/null || true
}

verify_sources=()
while IFS= read -r source; do
  verify_sources+=("$source")
done < <(
  find "$VERIFY_SOURCE" -maxdepth 1 -type d \
    \( -name "gemma4_12b_layer??_verify_seq${VERIFY_SEQ}_kv${CACHE_SIZE}_pal4_g16.mlpackage" \
       -o -name "gemma4_12b_layer??_verify_seq${VERIFY_SEQ}_kv${CACHE_SIZE}_pal4_g16.mlmodelc" \) \
    | sort
)

if [[ "${#verify_sources[@]}" -ne "$EXPECTED_LAYERS" ]]; then
  echo "expected $EXPECTED_LAYERS verify bundles, found ${#verify_sources[@]}" >&2
  exit 1
fi

for source in "${verify_sources[@]}"; do
  compile_or_copy "$source"
done
compile_or_copy "$DRAFTER_SOURCE"

python3 "$ROOT_DIR/scripts/verify_speculative_assets.py" \
  "$COMPILED_DIR" \
  --expected-layers "$EXPECTED_LAYERS" \
  --verify-seq "$VERIFY_SEQ" \
  --cache-size "$CACHE_SIZE" \
  --drafter-name "$DRAFTER_NAME"

while IFS= read -r compiled; do
  destination="$MODELS_DIR/$(basename "$compiled")"
  rm -rf "$destination"
  ditto --norsrc --noextattr "$compiled" "$destination"
  xattr -cr "$destination" 2>/dev/null || true
done < <(
  find "$COMPILED_DIR" -maxdepth 1 -type d \
    \( -name "gemma4_12b_layer??_verify_seq${VERIFY_SEQ}_kv${CACHE_SIZE}_pal4_g16.mlmodelc" \
       -o -name "${DRAFTER_NAME}.mlmodelc" \) \
    | sort
)

python3 "$ROOT_DIR/scripts/verify_speculative_assets.py" \
  "$MODELS_DIR" \
  --expected-layers "$EXPECTED_LAYERS" \
  --verify-seq "$VERIFY_SEQ" \
  --cache-size "$CACHE_SIZE" \
  --drafter-name "$DRAFTER_NAME"

echo "staged speculative assets in $MODELS_DIR"
