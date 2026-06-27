#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT="$ROOT_DIR/external/llama.cpp/examples/llama.swiftui/llama.swiftui.xcodeproj"
FRAMEWORK="$ROOT_DIR/external/llama.cpp/build-apple/llama.xcframework"
BUNDLE_ID="${BUNDLE_ID:-com.tasuku.Gemma12BProbe}"

if [[ ! -d "$FRAMEWORK" ]]; then
  echo "llama.xcframework is missing. Run scripts/build_llama_xcframework.sh first." >&2
  exit 1
fi

"$ROOT_DIR/scripts/apply_ios_probe_patch.sh"

xcodebuild \
  -project "$PROJECT" \
  -scheme llama.swiftui \
  -sdk iphoneos \
  -configuration Release \
  -destination "generic/platform=iOS" \
  CODE_SIGNING_ALLOWED=NO \
  CODE_SIGNING_REQUIRED=NO \
  CODE_SIGN_IDENTITY= \
  PRODUCT_BUNDLE_IDENTIFIER="$BUNDLE_ID" \
  build

