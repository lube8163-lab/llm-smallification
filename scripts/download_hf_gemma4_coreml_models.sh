#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ID="${HF_MODEL_REPO:-lube8163/gemma-4-12b-coreml-iphone-practical-chat}"
DOWNLOAD_DIR="${HF_MODEL_DOWNLOAD_DIR:-$ROOT_DIR/runpod-artifacts/huggingface/gemma-4-12b-coreml-iphone-practical-chat}"
MODEL_DIR="${DST_DIR:-$ROOT_DIR/ios/CoreMLProbe/CoreMLProbe/Models}"

if ! command -v hf >/dev/null 2>&1; then
  echo "missing Hugging Face CLI: install it from https://huggingface.co/docs/huggingface_hub/guides/cli" >&2
  exit 1
fi

echo "== download $REPO_ID =="
mkdir -p "$DOWNLOAD_DIR"
hf download "$REPO_ID" --repo-type model --local-dir "$DOWNLOAD_DIR"

if [[ ! -f "$DOWNLOAD_DIR/SHA256SUMS" ]]; then
  echo "missing checksum manifest: $DOWNLOAD_DIR/SHA256SUMS" >&2
  exit 1
fi

echo "== verify published package files =="
(
  cd "$DOWNLOAD_DIR"
  shasum -a 256 -c SHA256SUMS
)

if [[ "${DOWNLOAD_ONLY:-0}" == "1" ]]; then
  echo "download-only mode complete: $DOWNLOAD_DIR"
  exit 0
fi

if ! command -v xcrun >/dev/null 2>&1; then
  echo "missing xcrun; install Xcode or rerun with DOWNLOAD_ONLY=1" >&2
  exit 1
fi

echo "== compile Core ML packages =="
mkdir -p "$MODEL_DIR"
while IFS= read -r package; do
  name="$(basename "$package" .mlpackage)"
  output="$MODEL_DIR/$name.mlmodelc"
  if [[ -d "$output" && "${FORCE:-0}" != "1" ]]; then
    echo "skip existing: $name"
    continue
  fi
  rm -rf "$output"
  echo "compile: $name"
  xcrun coremlcompiler compile "$package" "$MODEL_DIR" >/dev/null
done < <(find "$DOWNLOAD_DIR/models" -type d -name '*.mlpackage' -prune | sort)

xattr -cr "$MODEL_DIR" 2>/dev/null || true
echo "ready: $MODEL_DIR"
