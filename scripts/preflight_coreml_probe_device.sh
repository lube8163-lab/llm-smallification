#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${DST_DIR:-"$ROOT_DIR/ios/CoreMLProbe/CoreMLProbe/Models"}"
EXPECTED_LAYERS="${EXPECTED_LAYERS:-48}"
SEQ_LEN="${SEQ_LEN:-4}"
MAX_DECODER_LAYERS_PER_BUNDLE="${MAX_DECODER_LAYERS_PER_BUNDLE:-4}"
if [[ -z "${MAX_PEAK_MB:-}" ]]; then
  if [[ "$SEQ_LEN" -gt 4 ]]; then
    MAX_PEAK_MB=900
  else
    MAX_PEAK_MB=350
  fi
fi
BUILD_DESTINATION="${BUILD_DESTINATION:-generic/platform=iOS}"
ALLOW_LEGACY_LM_HEAD=0
REQUIRE_IMAGE_EMBEDDER="${REQUIRE_IMAGE_EMBEDDER:-0}"
REQUIRE_AUDIO_EMBEDDER="${REQUIRE_AUDIO_EMBEDDER:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"

usage() {
  cat <<'EOF'
Usage:
  preflight_coreml_probe_device.sh [options]

Options:
  --allow-legacy-lm-head  Permit the old linear-only LM head for status checks.
  --require-image-embedder
                          Require the 32-patch image embedder bundle used by
                          image-smoke automation.
  --require-audio-embedder
                          Require the 32-token audio embedder bundle used by
                          audio-smoke automation.
  --skip-build            Verify assets only; do not run a generic iOS build.
  --destination X         Destination passed to build_coreml_probe_ios.sh.
  -h, --help              Show this help.

Environment:
  DST_DIR=PATH            Override the iOS Models directory.
  EXPECTED_LAYERS=N       Expected decoder layer count (default: 48).
  SEQ_LEN=N               Expected fixed sequence length (default: 4).
  MAX_DECODER_LAYERS_PER_BUNDLE=N
                          Maximum decoder layers per compiled bundle (default: 4).
  MAX_PEAK_MB=N           Suggested analyzer peak-memory ceiling (default: 350, or 900 when SEQ_LEN > 4).
  BUILD_DESTINATION=X     Default destination for the iOS build.
  REQUIRE_IMAGE_EMBEDDER=1
                          Same as --require-image-embedder.
  REQUIRE_AUDIO_EMBEDDER=1
                          Same as --require-audio-embedder.
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
    --require-image-embedder)
      REQUIRE_IMAGE_EMBEDDER=1
      shift
      ;;
    --require-audio-embedder)
      REQUIRE_AUDIO_EMBEDDER=1
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
echo "sequence length: $SEQ_LEN"
echo "max decoder layers per bundle: $MAX_DECODER_LAYERS_PER_BUNDLE"
echo "require image embedder: $REQUIRE_IMAGE_EMBEDDER"
echo "require audio embedder: $REQUIRE_AUDIO_EMBEDDER"

verify_args=(
  "$ROOT_DIR/scripts/verify_coreml_probe_assets.py"
  "$MODEL_DIR"
  --expected-layers "$EXPECTED_LAYERS"
  --seq-len "$SEQ_LEN"
  --max-decoder-layers-per-bundle "$MAX_DECODER_LAYERS_PER_BUNDLE"
)

if [[ "$ALLOW_LEGACY_LM_HEAD" == "1" ]]; then
  echo "mode: legacy LM head allowed"
  verify_args+=(--allow-legacy-lm-head)
else
  echo "mode: strict norm+lm_head required"
  verify_args+=(--require-norm-lm-head --fail-on-legacy)
fi

if [[ "$REQUIRE_IMAGE_EMBEDDER" == "1" ]]; then
  verify_args+=(--require-image-embedder)
fi

if [[ "$REQUIRE_AUDIO_EMBEDDER" == "1" ]]; then
  verify_args+=(--require-audio-embedder)
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
  decoder compute:  All
  layers:           First $EXPECTED_LAYERS
  window:           Seq $SEQ_LEN
  cache:            Run end
  tokens:           8 first, then 16 if peak memory remains stable
  retain decoders:  0 baseline; 1 is available only for comparison

After the Xcode run, save the console output and analyze it with:
  ./scripts/analyze_coreml_probe_log.py <log> --require-norm-lm-head --fail-on-repeat --expect-layers $EXPECTED_LAYERS --min-generated-tokens 8 --max-peak-mb $MAX_PEAK_MB
EOF
