#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_DIR="$ROOT_DIR/external/llama.cpp"
PATCH_FILE="$ROOT_DIR/patches/llama-swiftui-gemma12b-probe.patch"

if [[ ! -d "$LLAMA_DIR/.git" ]]; then
  echo "llama.cpp is missing. Run scripts/setup_llama_cpp.sh first." >&2
  exit 1
fi

if git -C "$LLAMA_DIR" apply --check "$PATCH_FILE" >/dev/null 2>&1; then
  git -C "$LLAMA_DIR" apply "$PATCH_FILE"
  echo "Applied $PATCH_FILE"
elif git -C "$LLAMA_DIR" apply --reverse --check "$PATCH_FILE" >/dev/null 2>&1; then
  echo "Patch already applied: $PATCH_FILE"
else
  echo "Patch did not apply cleanly and does not look already applied." >&2
  exit 1
fi
