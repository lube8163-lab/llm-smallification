#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT="$ROOT_DIR/ios/CoreMLProbe/CoreMLProbe.xcodeproj"
DERIVED_DATA="$ROOT_DIR/.derived-data/CoreMLProbe"
DESTINATION="${1:-platform=iOS Simulator,name=iPhone 17,OS=26.5}"

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

xattr -cr "$ROOT_DIR/ios/CoreMLProbe/CoreMLProbe" "$DERIVED_DATA" 2>/dev/null || true
if [[ -d "$ROOT_DIR/ios/CoreMLProbe/CoreMLProbe/Models" ]]; then
  find "$ROOT_DIR/ios/CoreMLProbe/CoreMLProbe/Models" -mindepth 0 -exec xattr -c {} + 2>/dev/null || true
  clear_packaging_xattrs "$ROOT_DIR/ios/CoreMLProbe/CoreMLProbe/Models"
fi
clear_packaging_xattrs "$DERIVED_DATA/Build/Products/Debug-iphoneos/CoreMLProbe.app"
clear_packaging_xattrs "$DERIVED_DATA/Build/Products/Debug-iphonesimulator/CoreMLProbe.app"

xcodebuild \
  -project "$PROJECT" \
  -scheme CoreMLProbe \
  -configuration Debug \
  -destination "$DESTINATION" \
  -derivedDataPath "$DERIVED_DATA" \
  build
