#!/usr/bin/env bash
# Publish only the measured speculative-decoding additions to the existing
# Hugging Face model repository. Existing practical-chat files remain intact.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ID="${HF_MODEL_REPO:-lube8163/gemma-4-12b-coreml-iphone-practical-chat}"
VERIFY_DIR="${HF_VERIFY_DIR:-$ROOT_DIR/runpod-artifacts/speculative/verify}"
DRAFTER_PACKAGE="${HF_DRAFTER_PACKAGE:-$ROOT_DIR/runpod-artifacts/speculative/drafter/gemma4_12b_drafter_step_kv512_mixed_pal4_g16_head_int8.mlpackage}"
FUSED_DIR="${HF_FUSED_DIR:-$HOME/Library/Caches/llm-smallification/CoreMLProbe-Fused6/packages/group00}"
CARD_DIR="$ROOT_DIR/docs/huggingface/gemma-4-12b-coreml-iphone-practical-chat"
REMOTE_SHA_URL="https://huggingface.co/$REPO_ID/raw/main/SHA256SUMS"

if ! command -v hf >/dev/null 2>&1; then
  echo "missing Hugging Face CLI" >&2
  exit 1
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "missing jq" >&2
  exit 1
fi

expected_drafter="gemma4_12b_drafter_step_kv512_mixed_pal4_g16_head_int8.mlpackage"
fused_names=(
  "gemma4_12b_layers00_05_prefill_seq320_kv_pal4_g16.mlpackage"
  "gemma4_12b_layers00_05_verify_seq4_kv512_pal4_g16.mlpackage"
)

[[ "$(basename "$DRAFTER_PACKAGE")" == "$expected_drafter" && -d "$DRAFTER_PACKAGE" ]] || {
  echo "missing or unexpected selected drafter: $DRAFTER_PACKAGE" >&2
  exit 1
}
[[ "$DRAFTER_PACKAGE" != *"backup-reversed-input-order"* ]] || {
  echo "refusing to publish reversed-input drafter backup" >&2
  exit 1
}

verify_packages=()
for layer in $(seq -w 0 47); do
  package="$VERIFY_DIR/gemma4_12b_layer${layer}_verify_seq4_kv512_pal4_g16.mlpackage"
  [[ -d "$package" ]] || {
    echo "missing verify package for layer $layer: $package" >&2
    exit 1
  }
  verify_packages+=("$package")
done
actual_verify_count="$(find "$VERIFY_DIR" -maxdepth 1 -type d -name '*.mlpackage' | wc -l | tr -d ' ')"
[[ "$actual_verify_count" == "48" ]] || {
  echo "expected exactly 48 verify packages, found $actual_verify_count" >&2
  exit 1
}

for name in "${fused_names[@]}"; do
  [[ -d "$FUSED_DIR/$name" ]] || {
    echo "missing selected fused package: $FUSED_DIR/$name" >&2
    exit 1
  }
done
if find "$FUSED_DIR" -maxdepth 1 -name '*layers06_11*' -print -quit | grep -q .; then
  echo "refusing source directory that contains the slower layers06_11 group" >&2
  exit 1
fi

jq -e '.release == "speculative-fused6"' "$CARD_DIR/OPTIONAL_ASSETS.json" >/dev/null

staging_dir="$(mktemp -d "${TMPDIR:-/tmp}/gemma4-hf-speculative.XXXXXX")"
cleanup() {
  rm -rf "$staging_dir"
}
trap cleanup EXIT

mkdir -p \
  "$staging_dir/models/speculative/verify" \
  "$staging_dir/models/speculative/drafter" \
  "$staging_dir/models/fused/kv"

echo "== stage portable Core ML packages =="
for package in "${verify_packages[@]}"; do
  cp -cR "$package" "$staging_dir/models/speculative/verify/"
done
cp -cR "$DRAFTER_PACKAGE" "$staging_dir/models/speculative/drafter/"
for name in "${fused_names[@]}"; do
  cp -cR "$FUSED_DIR/$name" "$staging_dir/models/fused/kv/"
done
cp "$CARD_DIR/README.md" "$staging_dir/README.md"
cp "$CARD_DIR/OPTIONAL_ASSETS.json" "$staging_dir/OPTIONAL_ASSETS.json"

echo "== update repository checksum manifest =="
curl -fsSL "$REMOTE_SHA_URL" -o "$staging_dir/SHA256SUMS.previous"
grep -Ev '  models/(speculative|fused)/' \
  "$staging_dir/SHA256SUMS.previous" >"$staging_dir/SHA256SUMS"

new_checksum_manifest="$staging_dir/SHA256SUMS.new"
(
  cd "$staging_dir"
  find models/speculative models/fused -type f | LC_ALL=C sort | xargs shasum -a 256
) >"$new_checksum_manifest"
cat "$new_checksum_manifest" >>"$staging_dir/SHA256SUMS"
LC_ALL=C sort -k2 "$staging_dir/SHA256SUMS" -o "$staging_dir/SHA256SUMS"
rm "$staging_dir/SHA256SUMS.previous"

(
  cd "$staging_dir"
  shasum -a 256 -c "$new_checksum_manifest"
)

package_count="$(find "$staging_dir/models" -type d -name '*.mlpackage' | wc -l | tr -d ' ')"
total_size="$(du -sh "$staging_dir/models" | awk '{print $1}')"
[[ "$package_count" == "51" ]] || {
  echo "expected 51 staged packages, found $package_count" >&2
  exit 1
}
echo "staged packages=$package_count size=$total_size"

if [[ "${HF_PUBLISH_DRY_RUN:-0}" == "1" ]]; then
  echo "dry run complete; authentication and upload skipped"
  exit 0
fi

hf auth whoami >/dev/null
code_commit="$(git -C "$ROOT_DIR" rev-parse HEAD)"
hf upload "$REPO_ID" "$staging_dir" . \
  --repo-type model \
  --commit-message "Add speculative decoding and selected fused6 models" \
  --commit-description "Portable Core ML additions validated on iPhone 14 and iPhone 17. Conversion code commit: $code_commit"

echo "published: https://huggingface.co/$REPO_ID"
