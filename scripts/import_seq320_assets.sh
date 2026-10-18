#!/usr/bin/env bash
# Pull the Seq320 pal4 decoder singles + int4 embedding from the RunPod pod,
# compile them, and stage into the iOS app Models directory.
#
# To make room in the app bundle, the legacy int4 seq64 decoder chunk bundles
# (the pre-pal4 A/B stack, ~5.9GB, unused by the default pal4 path) are moved
# to runpod-artifacts/retired-int4-seq64-chunks/ — kept on disk, not deleted.
#
# Usage: ./scripts/import_seq320_assets.sh <pod-ip> <pod-port>
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POD_IP="${1:?pod ip}"
POD_PORT="${2:?pod ssh port}"
KEY="$HOME/.ssh/runpod_tiny_image_model_ed25519"
MODELS="$ROOT_DIR/ios/CoreMLProbe/CoreMLProbe/Models"
STAGING="$ROOT_DIR/runpod-artifacts/seq320"
RETIRED="$ROOT_DIR/runpod-artifacts/retired-int4-seq64-chunks"

mkdir -p "$STAGING" "$RETIRED"

echo "== 1/4 transfer mlpackages from pod (single tar stream) =="
ssh -i "$KEY" -p "$POD_PORT" -o StrictHostKeyChecking=no "root@$POD_IP" \
  'tar cf - -C /workspace/gemma12b coreml-layers-seq320-pal4-singles coreml-endpoints-seq320-int4' \
  | tar xf - -C "$STAGING"

echo "== 2/4 compile to mlmodelc =="
COMPILED="$STAGING/compiled"
mkdir -p "$COMPILED"
for pkg in "$STAGING"/coreml-layers-seq320-pal4-singles/*.mlpackage \
           "$STAGING"/coreml-endpoints-seq320-int4/*.mlpackage; do
  [ -d "$pkg" ] || continue
  name="$(basename "$pkg" .mlpackage)"
  if [ -d "$COMPILED/$name.mlmodelc" ]; then
    echo "skip (compiled) $name"
    continue
  fi
  echo "compile $name"
  xcrun coremlcompiler compile "$pkg" "$COMPILED" >/dev/null
done

echo "== 3/4 retire legacy int4 seq64 decoder chunks from the app bundle =="
for chunk in "$MODELS"/gemma4_12b_layers*_decoder_seq64_mask_int4_block32.mlmodelc; do
  [ -d "$chunk" ] || continue
  echo "retire $(basename "$chunk")"
  mv "$chunk" "$RETIRED/"
done

echo "== 4/4 stage compiled seq320 bundles into Models =="
for compiled in "$COMPILED"/*.mlmodelc; do
  name="$(basename "$compiled")"
  rm -rf "$MODELS/$name"
  cp -R "$compiled" "$MODELS/$name"
done
xattr -cr "$MODELS" 2>/dev/null || true

echo "== done =="
du -sh "$MODELS"
ls "$MODELS" | grep -c seq320 || true
