#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_DIR="$ROOT_DIR/external/llama.cpp"
BUILD_DIR="$LLAMA_DIR/build"
JOBS="${JOBS:-4}"

mkdir -p "$ROOT_DIR/external"

if [[ ! -d "$LLAMA_DIR/.git" ]]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
fi

cmake -S "$LLAMA_DIR" -B "$BUILD_DIR" \
  -DGGML_METAL=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build "$BUILD_DIR" --target llama-cli -j "$JOBS"

"$BUILD_DIR/bin/llama-cli" --version

