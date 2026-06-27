#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_DIR="$ROOT_DIR/external/llama.cpp"

if [[ ! -d "$LLAMA_DIR/.git" ]]; then
  echo "llama.cpp is missing. Run scripts/setup_llama_cpp.sh first." >&2
  exit 1
fi

(
  cd "$LLAMA_DIR"
  ./build-xcframework.sh
)

echo "Built: $LLAMA_DIR/build-apple/llama.xcframework"

