#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${DST_DIR:-"$ROOT_DIR/ios/CoreMLProbe/CoreMLProbe/Models"}"
EXPECTED_LAYERS="${EXPECTED_LAYERS:-48}"
BUILD_DESTINATION="${BUILD_DESTINATION:-generic/platform=iOS}"
ALLOW_LEGACY_LM_HEAD=0
SKIP_BUILD="${SKIP_BUILD:-0}"

usage() {
  cat <<'EOF'
Usage:
  preflight_coreml_probe_device.sh [options]

Options:
  --allow-legacy-lm-head  Permit the old linear-only LM head for status checks.
  --skip-build            Verify assets only; do not run a generic iOS build.
  --destination X         Destination passed to build_coreml_probe_ios.sh.
  -h, --help              Show this help.

Environment:
  DST_DIR=PATH            Override the iOS Models directory.
  EXPECTED_LAYERS=N       Expected decoder layer count (default: 48).
  BUILD_DESTINATION=X     Default destination for the iOS build.
  SKIP_BUILD=1            Same as --skip-build.

By default this is a strict preflight for the next real-device run: it requires
the corrected norm+lm_head endpoint and fails if the legacy LM head is present.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --allow-legacy-lm-head)
      ALLOW_LEGACY_LM_HEAD=1
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
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

echo "CoreMLProbe device preflight"
echo "models: $MODEL_DIR"
echo "expected decoder layers: $EXPECTED_LAYERS"

verify_args=(
  "$ROOT_DIR/scripts/verify_coreml_probe_assets.py"
  "$MODEL_DIR"
  --expected-layers "$EXPECTED_LAYERS"
)

if [[ "$ALLOW_LEGACY_LM_HEAD" == "1" ]]; then
  echo "mode: legacy LM head allowed"
  verify_args+=(--allow-legacy-lm-head)
else
  echo "mode: strict norm+lm_head required"
  verify_args+=(--require-norm-lm-head --fail-on-legacy)
fi

"${verify_args[@]}"

if [[ "$SKIP_BUILD" == "1" ]]; then
  echo "skip build: SKIP_BUILD=1"
else
  "$ROOT_DIR/scripts/build_coreml_probe_ios.sh" "$BUILD_DESTINATION"
fi

cat <<EOF
device preflight passed

Recommended real-device settings:
  endpoint compute: CPU
  decoder compute:  CPU+GPU
  layers:           First $EXPECTED_LAYERS
  cache:            Run end
  tokens:           8 first, then 16 if peak memory remains stable

After the Xcode run, save the console output and analyze it with:
  ./scripts/analyze_coreml_probe_log.py <log> --require-norm-lm-head --fail-on-repeat --expect-layers $EXPECTED_LAYERS --min-generated-tokens 8 --max-peak-mb 350
EOF
