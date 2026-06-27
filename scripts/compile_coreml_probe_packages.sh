#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="${1:-"$ROOT_DIR/runpod-artifacts/coreml-probes"}"
DST_DIR="${2:-"$ROOT_DIR/runpod-artifacts/compiled"}"
FORCE="${FORCE:-0}"

mkdir -p "$DST_DIR"

packages=()
while IFS= read -r package; do
  packages+=("$package")
done < <(find "$SRC_DIR" -maxdepth 1 -type d -name "*.mlpackage" | sort)

if [[ "${#packages[@]}" -eq 0 ]]; then
  echo "missing: no .mlpackage directories under $SRC_DIR" >&2
  exit 1
fi

for package in "${packages[@]}"; do
  name="$(basename "$package" .mlpackage)"
  out="$DST_DIR/$name.mlmodelc"
  if [[ -d "$out" && "$FORCE" != "1" ]]; then
    echo "skip existing: $out"
    du -sh "$out"
    continue
  fi

  rm -rf "$out"
  echo "compile: $package"
  xcrun coremlcompiler compile "$package" "$DST_DIR"
  xattr -cr "$out" 2>/dev/null || true
  du -sh "$out"
done
