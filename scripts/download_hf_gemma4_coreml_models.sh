#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ID="${HF_MODEL_REPO:-lube8163/gemma-4-12b-coreml-iphone-practical-chat}"
DOWNLOAD_DIR="${HF_MODEL_DOWNLOAD_DIR:-$ROOT_DIR/runpod-artifacts/huggingface/gemma-4-12b-coreml-iphone-practical-chat}"
MODEL_DIR="${DST_DIR:-$ROOT_DIR/ios/CoreMLProbe/CoreMLProbe/Models}"
PROFILE="${HF_MODEL_PROFILE:-practical}"

if ! command -v hf >/dev/null 2>&1; then
  echo "missing Hugging Face CLI: install it from https://huggingface.co/docs/huggingface_hub/guides/cli" >&2
  exit 1
fi

case "$PROFILE" in
  practical)
    include_patterns=(
      "README.md"
      "LICENSE"
      "NOTICE.md"
      "SHA256SUMS"
      "models/endpoints/**"
      "models/kv/prefill/**"
      "models/kv/decode/**"
      "models/multimodal/**"
    )
    compile_roots=(
      "$DOWNLOAD_DIR/models/endpoints"
      "$DOWNLOAD_DIR/models/kv/prefill"
      "$DOWNLOAD_DIR/models/kv/decode"
      "$DOWNLOAD_DIR/models/multimodal"
    )
    ;;
  speculative)
    include_patterns=(
      "README.md"
      "LICENSE"
      "NOTICE.md"
      "SHA256SUMS"
      "models/**"
    )
    compile_roots=("$DOWNLOAD_DIR/models")
    ;;
  *)
    echo "unsupported HF_MODEL_PROFILE=$PROFILE (expected practical or speculative)" >&2
    exit 1
    ;;
esac

echo "== download $REPO_ID =="
mkdir -p "$DOWNLOAD_DIR"
download_args=(
  "$REPO_ID"
  --repo-type model
  --local-dir "$DOWNLOAD_DIR"
)
for pattern in "${include_patterns[@]}"; do
  download_args+=(--include "$pattern")
done
hf download "${download_args[@]}"

if [[ ! -f "$DOWNLOAD_DIR/SHA256SUMS" ]]; then
  echo "missing checksum manifest: $DOWNLOAD_DIR/SHA256SUMS" >&2
  exit 1
fi

echo "== verify published package files =="
verification_manifest="$(mktemp)"
trap 'rm -f "$verification_manifest"' EXIT
while read -r checksum relative_path; do
  if [[ -f "$DOWNLOAD_DIR/$relative_path" ]]; then
    printf '%s  %s\n' "$checksum" "$relative_path" >>"$verification_manifest"
  fi
done <"$DOWNLOAD_DIR/SHA256SUMS"

if [[ ! -s "$verification_manifest" ]]; then
  echo "checksum manifest has no entries for the downloaded profile" >&2
  exit 1
fi

while IFS= read -r downloaded_file; do
  relative_path="${downloaded_file#"$DOWNLOAD_DIR/"}"
  if ! grep -Fq "  $relative_path" "$DOWNLOAD_DIR/SHA256SUMS"; then
    echo "downloaded model file is absent from SHA256SUMS: $relative_path" >&2
    exit 1
  fi
done < <(find "${compile_roots[@]}" -type f 2>/dev/null | sort)

(
  cd "$DOWNLOAD_DIR"
  shasum -a 256 -c "$verification_manifest"
)

if [[ "${DOWNLOAD_ONLY:-0}" == "1" ]]; then
  echo "download-only mode complete: profile=$PROFILE path=$DOWNLOAD_DIR"
  exit 0
fi

if ! command -v xcrun >/dev/null 2>&1; then
  echo "missing xcrun; install Xcode or rerun with DOWNLOAD_ONLY=1" >&2
  exit 1
fi

echo "== compile Core ML packages =="
mkdir -p "$MODEL_DIR"
while IFS= read -r package; do
  name="$(basename "$package" .mlpackage)"
  output="$MODEL_DIR/$name.mlmodelc"
  if [[ -d "$output" && "${FORCE:-0}" != "1" ]]; then
    echo "skip existing: $name"
    continue
  fi
  rm -rf "$output"
  echo "compile: $name"
  xcrun coremlcompiler compile "$package" "$MODEL_DIR" >/dev/null
done < <(find "${compile_roots[@]}" -type d -name '*.mlpackage' -prune 2>/dev/null | sort)

xattr -cr "$MODEL_DIR" 2>/dev/null || true
echo "ready: profile=$PROFILE path=$MODEL_DIR"
