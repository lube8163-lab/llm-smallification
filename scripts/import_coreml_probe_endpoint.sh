#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPILED_MODE=0
SKIP_BUILD="${SKIP_BUILD:-0}"
BUILD_DESTINATION="${BUILD_DESTINATION:-generic/platform=iOS}"
EXPECTED_LAYERS="${EXPECTED_LAYERS:-48}"

usage() {
  cat <<'EOF'
Usage:
  import_coreml_probe_endpoint.sh [options] <endpoint-mlpackage-dir-or-tar.gz> [compiled-output-dir]
  import_coreml_probe_endpoint.sh --compiled [options] <compiled-endpoint-dir-or-mlmodelc>

Options:
  --compiled       Treat input as already compiled .mlmodelc bundles.
  --skip-build     Refresh and verify assets, but do not run a generic iOS build.
  --destination X  Destination passed to preflight_coreml_probe_device.sh.
  -h, --help       Show this help.

Environment:
  DST_DIR=PATH     Override the iOS Models directory.
  EXPECTED_LAYERS  Expected decoder layer count for verification (default: 48).
  BUILD_DESTINATION=X
                  Default destination for the generic iOS build.
  SKIP_BUILD=1     Same as --skip-build.

This is the Mac-side one-shot after copying the RunPod endpoint artifact back.
It refreshes the staged norm+lm_head endpoint, removes the stale legacy LM head,
then runs the strict real-device preflight once.
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

INPUT="$1"
COMPILED_DIR="${2:-}"

if [[ "$COMPILED_MODE" == "1" && $# -gt 1 ]]; then
  echo "--compiled accepts only one input path" >&2
  exit 2
fi

refresh_args=("$ROOT_DIR/scripts/refresh_coreml_probe_endpoint.sh" --skip-build)
if [[ "$COMPILED_MODE" == "1" ]]; then
  refresh_args+=(--compiled "$INPUT")
else
  refresh_args+=("$INPUT")
  if [[ -n "$COMPILED_DIR" ]]; then
    refresh_args+=("$COMPILED_DIR")
  fi
fi

"${refresh_args[@]}"

preflight_args=("$ROOT_DIR/scripts/preflight_coreml_probe_device.sh")
if [[ "$SKIP_BUILD" == "1" ]]; then
  preflight_args+=(--skip-build)
else
  preflight_args+=(--destination "$BUILD_DESTINATION")
fi

"${preflight_args[@]}"

cat <<EOF
endpoint import complete

Next Xcode run:
  endpoint compute: CPU
  decoder compute:  CPU+GPU
  layers:           First $EXPECTED_LAYERS
  cache:            Run end
  tokens:           8 first
EOF
