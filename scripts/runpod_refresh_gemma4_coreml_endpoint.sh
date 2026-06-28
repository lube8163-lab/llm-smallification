#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="/workspace/gemma12b/models/gemma-4-12b-it-qat-q4_0-unquantized"
OUT_DIR="/workspace/gemma12b/coreml-endpoints-seq4-int4"
ARCHIVE="/workspace/gemma12b/gemma4-norm-lm-head-endpoint.tar.gz"
SEQ_LEN=4
BLOCK_SIZE=32
FORCE_CONVERT=1
KEEP_FP16=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  runpod_refresh_gemma4_coreml_endpoint.sh [options]

Options:
  --model-dir PATH   Gemma QAT-unquantized model directory.
  --out-dir PATH     Core ML endpoint .mlpackage output directory.
  --archive PATH     Transfer archive to write after conversion.
  --seq-len N        Embedding sequence length for optional embedding target.
  --block-size N     Int4 block size.
  --no-force         Skip conversion when the target package already exists.
  --keep-fp16        Also save fp16 .mlpackage output.
  --dry-run          Print commands without executing them.
  -h, --help         Show this help.

This RunPod-side wrapper converts the corrected norm+lm_head endpoint and then
writes SHA256SUMS plus a transfer tarball. Run it from the repo checkout on a
CUDA host with the Gemma weights available.
EOF
}

require_value() {
  if [[ $# -lt 2 || "${2:-}" == --* ]]; then
    echo "missing value for $1" >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-dir)
      require_value "$@"
      MODEL_DIR="$2"
      shift 2
      ;;
    --out-dir)
      require_value "$@"
      OUT_DIR="$2"
      shift 2
      ;;
    --archive)
      require_value "$@"
      ARCHIVE="$2"
      shift 2
      ;;
    --seq-len)
      require_value "$@"
      SEQ_LEN="$2"
      shift 2
      ;;
    --block-size)
      require_value "$@"
      BLOCK_SIZE="$2"
      shift 2
      ;;
    --no-force)
      FORCE_CONVERT=0
      shift
      ;;
    --keep-fp16)
      KEEP_FP16=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

run_cmd() {
  printf '+'
  printf ' %s' "$@"
  printf '\n'
  if [[ "$DRY_RUN" != "1" ]]; then
    "$@"
  fi
}

if [[ "$DRY_RUN" != "1" && ! -d "$MODEL_DIR" ]]; then
  echo "missing model dir: $MODEL_DIR" >&2
  exit 1
fi

convert_args=(
  python
  "$ROOT_DIR/scripts/runpod_convert_gemma4_coreml_endpoints.py"
  --target norm-lm-head
  --model-dir "$MODEL_DIR"
  --out-dir "$OUT_DIR"
  --seq-len "$SEQ_LEN"
  --block-size "$BLOCK_SIZE"
)

if [[ "$FORCE_CONVERT" == "1" ]]; then
  convert_args+=(--force)
fi
if [[ "$KEEP_FP16" == "1" ]]; then
  convert_args+=(--keep-fp16)
fi

run_cmd "${convert_args[@]}"
run_cmd python "$ROOT_DIR/scripts/package_runpod_coreml_endpoint.py" \
  --out-dir "$OUT_DIR" \
  --tar-gz "$ARCHIVE"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "RunPod endpoint refresh dry run complete: $ARCHIVE"
else
  echo "RunPod endpoint refresh complete: $ARCHIVE"
fi
