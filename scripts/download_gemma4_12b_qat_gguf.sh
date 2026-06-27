#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$ROOT_DIR/models/google-gemma-4-12b-qat-q4_0"
MODEL_FILE="gemma-4-12b-it-qat-q4_0.gguf"
MODEL_URL="https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf/resolve/main/$MODEL_FILE"
EXPECTED_BYTES=6975877728

mkdir -p "$MODEL_DIR"

OUT="$MODEL_DIR/$MODEL_FILE"
PART="$OUT.part"

if [[ -f "$OUT" ]]; then
  actual="$(wc -c < "$OUT" | tr -d ' ')"
  if [[ "$actual" == "$EXPECTED_BYTES" ]]; then
    echo "Model already present: $OUT"
    exit 0
  fi
  echo "Existing model has unexpected size: $actual bytes; expected $EXPECTED_BYTES" >&2
  exit 1
fi

wget -c --tries=20 --timeout=30 --progress=dot:giga -O "$PART" "$MODEL_URL"

actual="$(wc -c < "$PART" | tr -d ' ')"
if [[ "$actual" != "$EXPECTED_BYTES" ]]; then
  echo "Partial model has unexpected size: $actual bytes; expected $EXPECTED_BYTES" >&2
  exit 1
fi

mv "$PART" "$OUT"
echo "Downloaded: $OUT"

