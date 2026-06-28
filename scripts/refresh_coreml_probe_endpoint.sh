#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NORM_PACKAGE="gemma4_12b_norm_lm_head_1tok_int4_block32.mlpackage"
NORM_COMPILED="gemma4_12b_norm_lm_head_1tok_int4_block32.mlmodelc"
COMPILED_MODE=0
SKIP_BUILD="${SKIP_BUILD:-0}"
BUILD_DESTINATION="${BUILD_DESTINATION:-generic/platform=iOS}"
EXPECTED_LAYERS="${EXPECTED_LAYERS:-48}"
MODEL_DIR="${DST_DIR:-"$ROOT_DIR/ios/CoreMLProbe/CoreMLProbe/Models"}"

usage() {
  cat <<'EOF'
Usage:
  refresh_coreml_probe_endpoint.sh [options] <endpoint-mlpackage-dir> [compiled-output-dir]
  refresh_coreml_probe_endpoint.sh [options] <gemma4_12b_norm_lm_head_1tok_int4_block32.mlpackage> [compiled-output-dir]
  refresh_coreml_probe_endpoint.sh [options] <gemma4-norm-lm-head-endpoint.tar.gz> [compiled-output-dir]
  refresh_coreml_probe_endpoint.sh --compiled [options] <compiled-endpoint-dir>
  refresh_coreml_probe_endpoint.sh --compiled [options] <gemma4_12b_norm_lm_head_1tok_int4_block32.mlmodelc>

Options:
  --compiled       Treat the input directory as already compiled .mlmodelc bundles.
  --skip-build     Copy and verify assets, but do not run the iOS build.
  --destination X  Destination passed to build_coreml_probe_ios.sh.
  -h, --help       Show this help.

Environment:
  FORCE=1          Recompile endpoint package by default; set FORCE=0 to keep existing output.
  DST_DIR=PATH     Override the iOS Models directory used for copy/verify.
  EXPECTED_LAYERS  Expected decoder layer count for verification (default: 48).
  SKIP_BUILD=1     Same as --skip-build.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --compiled)
      COMPILED_MODE=1
      shift
      ;;
    --skip-build)
      SKIP_BUILD=1
      shift
      ;;
    --destination)
      if [[ $# -lt 2 ]]; then
        echo "missing value for --destination" >&2
        exit 2
      fi
      BUILD_DESTINATION="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      break
      ;;
  esac
done

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

INPUT_DIR="$1"
COMPILED_DIR="${2:-"$ROOT_DIR/runpod-artifacts/compiled-endpoints"}"
TMP_DIR=""

cleanup() {
  if [[ -n "$TMP_DIR" ]]; then
    rm -rf "$TMP_DIR"
  fi
}
trap cleanup EXIT

if [[ "$COMPILED_MODE" == "1" && $# -gt 1 ]]; then
  echo "--compiled accepts only one input directory" >&2
  exit 2
fi

if [[ "$COMPILED_MODE" == "1" ]]; then
  if [[ ! -d "$INPUT_DIR" ]]; then
    echo "missing compiled input directory: $INPUT_DIR" >&2
    exit 1
  fi
  if [[ "$(basename "$INPUT_DIR")" == "$NORM_COMPILED" ]]; then
    COMPILED_DIR="$(dirname "$INPUT_DIR")"
  else
    COMPILED_DIR="$INPUT_DIR"
  fi
  if [[ ! -d "$COMPILED_DIR/$NORM_COMPILED" ]]; then
    echo "missing compiled norm LM head: $COMPILED_DIR/$NORM_COMPILED" >&2
    exit 1
  fi
else
  case "$INPUT_DIR" in
    *.tar.gz|*.tgz)
      if [[ ! -f "$INPUT_DIR" ]]; then
        echo "missing endpoint archive: $INPUT_DIR" >&2
        exit 1
      fi
      TMP_DIR="$(mktemp -d)"
      tar -xzf "$INPUT_DIR" -C "$TMP_DIR"
      extracted_package="$(find "$TMP_DIR" -type d -name "$NORM_PACKAGE" -print -quit)"
      if [[ -z "$extracted_package" ]]; then
        echo "archive does not contain $NORM_PACKAGE: $INPUT_DIR" >&2
        exit 1
      fi
      INPUT_DIR="$(dirname "$extracted_package")"
      ;;
  esac

  if [[ ! -d "$INPUT_DIR" ]]; then
    echo "missing endpoint input directory: $INPUT_DIR" >&2
    exit 1
  fi
  if [[ "$(basename "$INPUT_DIR")" == "$NORM_PACKAGE" ]]; then
    INPUT_DIR="$(dirname "$INPUT_DIR")"
  fi
  if [[ ! -d "$INPUT_DIR/$NORM_PACKAGE" ]]; then
    echo "missing norm LM head package: $INPUT_DIR/$NORM_PACKAGE" >&2
    exit 1
  fi
  FORCE="${FORCE:-1}" "$ROOT_DIR/scripts/compile_coreml_probe_packages.sh" "$INPUT_DIR" "$COMPILED_DIR"
  if [[ ! -d "$COMPILED_DIR/$NORM_COMPILED" ]]; then
    echo "compile did not produce: $COMPILED_DIR/$NORM_COMPILED" >&2
    exit 1
  fi
fi

DST_DIR="$MODEL_DIR" "$ROOT_DIR/scripts/prepare_ios_coreml_probe_assets.sh" --endpoints-only "$COMPILED_DIR"
"$ROOT_DIR/scripts/verify_coreml_probe_assets.py" "$MODEL_DIR" \
  --expected-layers "$EXPECTED_LAYERS" \
  --require-norm-lm-head \
  --fail-on-legacy

if [[ "$SKIP_BUILD" != "1" ]]; then
  "$ROOT_DIR/scripts/build_coreml_probe_ios.sh" "$BUILD_DESTINATION"
fi

echo "refreshed norm LM head endpoint in $MODEL_DIR"
