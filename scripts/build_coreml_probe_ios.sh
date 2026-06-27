#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT="$ROOT_DIR/ios/CoreMLProbe/CoreMLProbe.xcodeproj"
DERIVED_DATA="$ROOT_DIR/.derived-data/CoreMLProbe"
DESTINATION="${1:-platform=iOS Simulator,name=iPhone 17,OS=26.5}"

xattr -cr "$ROOT_DIR/ios/CoreMLProbe/CoreMLProbe" "$DERIVED_DATA" 2>/dev/null || true

xcodebuild \
  -project "$PROJECT" \
  -scheme CoreMLProbe \
  -configuration Debug \
  -destination "$DESTINATION" \
  -derivedDataPath "$DERIVED_DATA" \
  build
