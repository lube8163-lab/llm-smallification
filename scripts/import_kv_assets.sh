#!/usr/bin/env bash
# Pull the KV-cache prefill/decode layer packages + seq1 embedding from the
# RunPod pod, compile them, and stage into the iOS app Models directory.
#
# The non-KV Seq320 pal4 decoder singles (48 bundles, ~5.3GB) become redundant
# once KV is validated, so they are moved to
# runpod-artifacts/retired-seq320-nonkv/ (kept on disk, not deleted). Pass
# --keep-nonkv to leave them in place for A/B.
#
# Usage: ./scripts/import_kv_assets.sh <pod-ip> <pod-port> [--keep-nonkv]
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POD_IP="${1:?pod ip}"
POD_PORT="${2:?pod ssh port}"
KEEP_NONKV="${3:-}"
KEY="$HOME/.ssh/runpod_tiny_image_model_ed25519"
MODELS="$ROOT_DIR/ios/CoreMLProbe/CoreMLProbe/Models"
STAGING="$ROOT_DIR/runpod-artifacts/kv"
RETIRED="$ROOT_DIR/runpod-artifacts/retired-seq320-nonkv"

mkdir -p "$STAGING" "$RETIRED"

echo "== 1/4 transfer mlpackages from pod =="
ssh -i "$KEY" -p "$POD_PORT" -o StrictHostKeyChecking=no "root@$POD_IP" \
  'tar cf - -C /workspace/gemma12b coreml-layers-kv-pal4 coreml-endpoints-seq1-int4' \
  | tar xf - -C "$STAGING"

echo "== 2/4 compile to mlmodelc =="
COMPILED="$STAGING/compiled"
mkdir -p "$COMPILED"
for pkg in "$STAGING"/coreml-layers-kv-pal4/*.mlpackage \
           "$STAGING"/coreml-endpoints-seq1-int4/*.mlpackage; do
  [ -d "$pkg" ] || continue
  name="$(basename "$pkg" .mlpackage)"
  if [ -d "$COMPILED/$name.mlmodelc" ]; then
    echo "skip (compiled) $name"; continue
  fi
  echo "compile $name"
  xcrun coremlcompiler compile "$pkg" "$COMPILED" >/dev/null
done

if [ "$KEEP_NONKV" != "--keep-nonkv" ]; then
  echo "== 3/4 retire non-KV Seq320 pal4 decoder singles =="
  for layer in "$MODELS"/gemma4_12b_layer*_decoder_seq320_mask_pal4_g16.mlmodelc; do
    [ -d "$layer" ] || continue
    echo "retire $(basename "$layer")"
    mv "$layer" "$RETIRED/"
  done
else
  echo "== 3/4 keeping non-KV Seq320 decoders (A/B mode) =="
fi

echo "== 4/4 stage compiled KV bundles into Models =="
for compiled in "$COMPILED"/*.mlmodelc; do
  name="$(basename "$compiled")"
  rm -rf "$MODELS/$name"
  cp -R "$compiled" "$MODELS/$name"
done
xattr -cr "$MODELS" 2>/dev/null || true

echo "== done =="
du -sh "$MODELS"
echo "prefill=$(ls "$MODELS" | grep -c prefill) decode=$(ls "$MODELS" | grep -c 'decode_kv') emb_seq1=$(ls "$MODELS" | grep -c embedding_seq1)"
