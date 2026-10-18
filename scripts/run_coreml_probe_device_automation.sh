#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Keep in sync with build_coreml_probe_ios.sh: outside iCloud (see comment there).
DERIVED_DATA="${COREML_PROBE_DERIVED_DATA:-$HOME/Library/Caches/llm-smallification/CoreMLProbe}"
APP_PATH="$DERIVED_DATA/Build/Products/Debug-iphoneos/CoreMLProbe.app"
BUNDLE_ID="${BUNDLE_ID:-lab.lube8163.CoreMLProbe}"

KIND="${COREML_PROBE_AUTOMATION_KIND:-chat}"
DEVICE="${COREML_PROBE_DEVICE:-}"
SKIP_BUILD=0
SKIP_INSTALL=0
NO_ANALYZE=0
REQUIRE_SPECULATIVE=0
MIN_SPECULATIVE_TOKENS_PER_SWEEP="${MIN_SPECULATIVE_TOKENS_PER_SWEEP:-}"
RUN_TIMEOUT="${COREML_PROBE_RUN_TIMEOUT:-1800}"
INSTALL_TIMEOUT="${COREML_PROBE_INSTALL_TIMEOUT:-1800}"
BUILD_DESTINATION="${BUILD_DESTINATION:-generic/platform=iOS}"
SEQ_LEN="${SEQ_LEN:-64}"
EXPECTED_LAYERS="${EXPECTED_LAYERS:-48}"
MAX_PEAK_MB="${MAX_PEAK_MB:-900}"
MIN_GENERATED_TOKENS="${MIN_GENERATED_TOKENS:-}"
TOKENS="${COREML_PROBE_GENERATE_TOKENS:-}"
RUN_MODE="${COREML_PROBE_MODE:-}"
RETAIN="${COREML_PROBE_RETAIN_DECODERS:-0}"
PROMPT="${COREML_PROBE_CHAT_PROMPT:-こんにちは。短く答えてください。}"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="${COREML_PROBE_AUTOMATION_LOG_DIR:-$DERIVED_DATA/DeviceAutomation/$TIMESTAMP}"

usage() {
  cat <<'EOF'
Usage:
  run_coreml_probe_device_automation.sh [options]

Options:
  --kind KIND             chat, probe, stability, retention, speed, tokens32,
                          image, audio, or multimodal (default: chat).
  --device ID             CoreDevice identifier, UDID, serial number, or name.
                          Defaults to the first connected iPhone.
  --tokens N              Generated-token count for chat/probe runs.
  --prompt TEXT           Chat prompt for --kind chat.
  --timeout SECONDS       devicectl launch timeout (default: 1800).
  --max-peak-mb N         Analyzer peak-memory ceiling (default: 900).
  --min-generated-tokens N
                          Analyzer generated-token floor. Defaults to 1 for
                          chat, token count for probe/tokens32, and 8 for sweeps.
  --require-speculative   Fail analysis unless the MTP path ran.
  --min-speculative-tokens-per-sweep N
                          Fail when verified output per target sweep is below N.
  --log-dir PATH          Output directory for build/install/launch/analyzer logs.
  --skip-build            Reuse the existing Debug-iphoneos app.
  --skip-install          Do not install before launching.
  --no-analyze            Skip analyze_coreml_probe_log.py.
  -h, --help              Show this help.

Environment:
  BUNDLE_ID                         App bundle identifier.
  COREML_PROBE_DEVICE               Default --device value.
  COREML_PROBE_AUTOMATION_KIND      Default --kind value.
  COREML_PROBE_CHAT_PROMPT          Default --prompt value.
  COREML_PROBE_GENERATE_TOKENS      Default --tokens value.
  COREML_PROBE_DISABLE_SPECULATIVE  Set to 1 for a seq-1 KV baseline run.
  COREML_PROBE_DRAFTER_MODEL        Override the bundled MTP drafter name.
  COREML_PROBE_DRAFTER_COMPUTE      Drafter compute units override.
  COREML_PROBE_SPECULATIVE_TREE     Set to 0 for linear-only speculative A/B.
  COREML_PROBE_KV_FULL_ANE          Set to 0/1 to override full-attention routing.
  COREML_PROBE_FUSED_COMPUTE        Compute units for fused KV groups.
  COREML_PROBE_RETAIN_VERIFY        Resident verify-layer count (diagnostic; default 0).
  ENDPOINT_COMPUTE                  Endpoint compute units. Defaults to All for
                                    pal4 and CPU for int4_block32.
  COREML_PROBE_RUN_TIMEOUT          Default --timeout value.
  COREML_PROBE_INSTALL_TIMEOUT      App-install timeout (default: 1800 seconds).
  COREML_PROBE_AUTOMATION_LOG_DIR   Default --log-dir value.
  BUILD_DESTINATION                 xcodebuild destination (default: generic/platform=iOS).
  SEQ_LEN, EXPECTED_LAYERS, MAX_PEAK_MB, MIN_GENERATED_TOKENS
                                    Analyzer and launch defaults.

The app is launched with automation environment variables and COREML_PROBE_AUTO_EXIT=1,
so devicectl --console returns after the run finishes.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --kind)
      KIND="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --tokens)
      TOKENS="$2"
      shift 2
      ;;
    --retain)
      RETAIN="$2"
      shift 2
      ;;
    --prompt)
      PROMPT="$2"
      shift 2
      ;;
    --timeout)
      RUN_TIMEOUT="$2"
      shift 2
      ;;
    --max-peak-mb)
      MAX_PEAK_MB="$2"
      shift 2
      ;;
    --min-generated-tokens)
      MIN_GENERATED_TOKENS="$2"
      shift 2
      ;;
    --require-speculative)
      REQUIRE_SPECULATIVE=1
      shift
      ;;
    --min-speculative-tokens-per-sweep)
      MIN_SPECULATIVE_TOKENS_PER_SWEEP="$2"
      REQUIRE_SPECULATIVE=1
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --skip-build)
      SKIP_BUILD=1
      shift
      ;;
    --skip-install)
      SKIP_INSTALL=1
      shift
      ;;
    --no-analyze)
      NO_ANALYZE=1
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

case "$KIND" in
  chat|chat-smoke)
    AUTORUN_KIND="chat"
    TOKENS="${TOKENS:-8}"
    MIN_GENERATED_TOKENS="${MIN_GENERATED_TOKENS:-1}"
    RUN_MODE="${RUN_MODE:-generate-token-loop}"
    ;;
  probe|run|single)
    AUTORUN_KIND="probe"
    TOKENS="${TOKENS:-8}"
    MIN_GENERATED_TOKENS="${MIN_GENERATED_TOKENS:-$TOKENS}"
    RUN_MODE="${RUN_MODE:-generate-token-loop}"
    ;;
  tokens32|ceiling32)
    AUTORUN_KIND="probe"
    TOKENS="${TOKENS:-32}"
    MIN_GENERATED_TOKENS="${MIN_GENERATED_TOKENS:-8}"
    RUN_MODE="${RUN_MODE:-generate-token-loop}"
    ;;
  stability|stability-sweep)
    AUTORUN_KIND="stability-sweep"
    TOKENS="${TOKENS:-8}"
    MIN_GENERATED_TOKENS="${MIN_GENERATED_TOKENS:-8}"
    RUN_MODE="${RUN_MODE:-generate-token-loop}"
    ;;
  retention|retain|retention-sweep|retain-sweep)
    AUTORUN_KIND="retention-sweep"
    TOKENS="${TOKENS:-8}"
    MIN_GENERATED_TOKENS="${MIN_GENERATED_TOKENS:-8}"
    RUN_MODE="${RUN_MODE:-generate-token-loop}"
    ;;
  speed|speed-sweep)
    AUTORUN_KIND="speed-sweep"
    TOKENS="${TOKENS:-4}"
    MIN_GENERATED_TOKENS="${MIN_GENERATED_TOKENS:-1}"
    RUN_MODE="${RUN_MODE:-generate-token-loop}"
    ;;
  image|image-smoke|vision-smoke|multimodal)
    AUTORUN_KIND="image-smoke"
    TOKENS="${TOKENS:-1}"
    MIN_GENERATED_TOKENS="${MIN_GENERATED_TOKENS:-1}"
    RUN_MODE="${RUN_MODE:-image-smoke}"
    ;;
  audio|audio-smoke)
    AUTORUN_KIND="audio-smoke"
    TOKENS="${TOKENS:-1}"
    MIN_GENERATED_TOKENS="${MIN_GENERATED_TOKENS:-1}"
    RUN_MODE="${RUN_MODE:-audio-smoke}"
    ;;
  multimodal-smoke|synthetic-multimodal)
    AUTORUN_KIND="multimodal-smoke"
    TOKENS="${TOKENS:-1}"
    MIN_GENERATED_TOKENS="${MIN_GENERATED_TOKENS:-1}"
    RUN_MODE="${RUN_MODE:-multimodal-smoke}"
    ;;
  memory-ramp|memory|ramp|mem-ramp)
    AUTORUN_KIND="memory-ramp"
    TOKENS="${TOKENS:-1}"
    MIN_GENERATED_TOKENS="${MIN_GENERATED_TOKENS:-0}"
    RUN_MODE="${RUN_MODE:-memory-ramp}"
    # The ramp intentionally holds many chunks resident (multi-GB peak) and
    # generates no tokens, so the token/peak analyzer does not apply.
    NO_ANALYZE=1
    MAX_PEAK_MB="${MAX_PEAK_MB_OVERRIDE:-8192}"
    ;;
  *)
    echo "unknown automation kind: $KIND" >&2
    usage >&2
    exit 2
    ;;
esac

mkdir -p "$LOG_DIR"
BUILD_LOG="$LOG_DIR/build.log"
INSTALL_LOG="$LOG_DIR/install.log"
LAUNCH_LOG="$LOG_DIR/device-console.log"
ANALYZE_LOG="$LOG_DIR/analyze.log"
DEVICES_JSON="$LOG_DIR/devices.json"
INSTALL_JSON="$LOG_DIR/install.json"
LAUNCH_JSON="$LOG_DIR/launch.json"

find_connected_iphone() {
  xcrun devicectl list devices --json-output "$DEVICES_JSON" >/dev/null
  python3 - "$DEVICES_JSON" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    devices = json.load(handle)["result"]["devices"]

def is_connected_iphone(device):
    hardware = device.get("hardwareProperties", {})
    device_properties = device.get("deviceProperties", {})
    connection = device.get("connectionProperties", {})
    return (
        hardware.get("platform") == "iOS"
        and hardware.get("deviceType") == "iPhone"
        and (
            connection.get("tunnelState") == "connected"
            or device_properties.get("ddiServicesAvailable") is True
        )
    )

for device in devices:
    if is_connected_iphone(device):
        print(device["identifier"])
        sys.exit(0)

print("no connected iPhone found", file=sys.stderr)
sys.exit(1)
PY
}

if [[ -z "$DEVICE" ]]; then
  DEVICE="$(find_connected_iphone)"
fi

echo "CoreMLProbe device automation"
echo "kind: $AUTORUN_KIND"
echo "device: $DEVICE"
echo "bundle: $BUNDLE_ID"
echo "logs: $LOG_DIR"
echo "seq: $SEQ_LEN"
echo "layers: first-$EXPECTED_LAYERS"
echo "tokens: $TOKENS"
echo "retain: $RETAIN"

if [[ "$SKIP_BUILD" == "1" ]]; then
  echo "skip build: reusing $APP_PATH"
else
  "$ROOT_DIR/scripts/build_coreml_probe_ios.sh" "$BUILD_DESTINATION" 2>&1 | tee "$BUILD_LOG"
fi

if [[ ! -d "$APP_PATH" ]]; then
  echo "missing app bundle: $APP_PATH" >&2
  exit 1
fi

if [[ "$SKIP_INSTALL" == "1" ]]; then
  echo "skip install"
else
  xcrun devicectl \
    --timeout "$INSTALL_TIMEOUT" \
    --json-output "$INSTALL_JSON" \
    device install app \
    --device "$DEVICE" \
    "$APP_PATH" 2>&1 | tee "$INSTALL_LOG"
fi

export COREML_PROBE_AUTORUN="$AUTORUN_KIND"
export COREML_PROBE_AUTO_EXIT=1
if [[ -n "${ENDPOINT_COMPUTE:-}" ]]; then
  export COREML_PROBE_ENDPOINT_COMPUTE="$ENDPOINT_COMPUTE"
elif [[ "${ENDPOINT_VARIANT:-}" == "int4_block32" ]]; then
  # The int4 endpoint can exceed the iPhone high-water limit while Core ML
  # compiles it for accelerators. Preserve the known-safe CPU path.
  export COREML_PROBE_ENDPOINT_COMPUTE=cpuOnly
else
  # The bundled pal4 norm+lm_head is ANE-compatible. CPU-only took 3-8s per
  # prediction on iPhone 14, versus ~20ms with Core ML free to select the ANE.
  export COREML_PROBE_ENDPOINT_COMPUTE=all
fi
export COREML_PROBE_DECODER_COMPUTE="${DECODER_COMPUTE:-all}"
if [[ -n "${DECODER_VARIANT:-}" ]]; then
  export COREML_PROBE_DECODER_VARIANT="$DECODER_VARIANT"
fi
if [[ -n "${ENDPOINT_VARIANT:-}" ]]; then
  export COREML_PROBE_ENDPOINT_VARIANT="$ENDPOINT_VARIANT"
fi
if [[ -n "${KEEP_E5_CACHE:-}" ]]; then
  export COREML_PROBE_KEEP_E5_CACHE="$KEEP_E5_CACHE"
fi
if [[ -n "${WARM_CHUNK:-}" ]]; then
  export COREML_PROBE_WARM_CHUNK="$WARM_CHUNK"
fi
if [[ -n "${GPU_CHUNKS:-}" ]]; then
  export COREML_PROBE_GPU_CHUNKS="$GPU_CHUNKS"
fi
if [[ -n "${WARM_MODEL:-}" ]]; then
  export COREML_PROBE_WARM_MODEL="$WARM_MODEL"
fi
export COREML_PROBE_MODE="$RUN_MODE"
export COREML_PROBE_LAYERS="first-$EXPECTED_LAYERS"
export COREML_PROBE_CACHE_POLICY=run-end-only
export COREML_PROBE_SEQ_LEN="$SEQ_LEN"
export COREML_PROBE_GENERATE_TOKENS="$TOKENS"
export COREML_PROBE_RETAIN_DECODERS="$RETAIN"
export COREML_PROBE_CHAT_PROMPT="$PROMPT"

ENV_JSON="$(
  python3 - <<'PY'
import json
import os

keys = [
    "COREML_PROBE_AUTORUN",
    "COREML_PROBE_AUTO_EXIT",
    "COREML_PROBE_ENDPOINT_COMPUTE",
    "COREML_PROBE_DECODER_COMPUTE",
    "COREML_PROBE_MODE",
    "COREML_PROBE_LAYERS",
    "COREML_PROBE_CACHE_POLICY",
    "COREML_PROBE_SEQ_LEN",
    "COREML_PROBE_GENERATE_TOKENS",
    "COREML_PROBE_DISABLE_SPECULATIVE",
    "COREML_PROBE_DRAFTER_MODEL",
    "COREML_PROBE_DRAFTER_COMPUTE",
    "COREML_PROBE_SPECULATIVE_TREE",
    "COREML_PROBE_KV_FULL_ANE",
    "COREML_PROBE_FUSED_COMPUTE",
    "COREML_PROBE_RETAIN_VERIFY",
    "COREML_PROBE_RETAIN_DECODERS",
    "COREML_PROBE_DECODER_VARIANT",
    "COREML_PROBE_ENDPOINT_VARIANT",
    "COREML_PROBE_KEEP_E5_CACHE",
    "COREML_PROBE_WARM_CHUNK",
    "COREML_PROBE_GPU_CHUNKS",
    "COREML_PROBE_WARM_MODEL",
    "COREML_PROBE_CHAT_PROMPT",
]
print(json.dumps({key: os.environ[key] for key in keys if key in os.environ}))
PY
)"

set +e
xcrun devicectl \
  --timeout "$RUN_TIMEOUT" \
  --json-output "$LAUNCH_JSON" \
  device process launch \
  --device "$DEVICE" \
  --environment-variables "$ENV_JSON" \
  --terminate-existing \
  --console \
  "$BUNDLE_ID" 2>&1 | tee "$LAUNCH_LOG"
launch_status=${PIPESTATUS[0]}
set -e

if [[ "$NO_ANALYZE" != "1" ]]; then
  analyzer_args=(
    --require-norm-lm-head
    --fail-on-repeat
    --expect-layers "$EXPECTED_LAYERS"
    --min-generated-tokens "$MIN_GENERATED_TOKENS"
    --max-peak-mb "$MAX_PEAK_MB"
  )
  if [[ "$REQUIRE_SPECULATIVE" == "1" ]]; then
    analyzer_args+=(--require-speculative)
  fi
  if [[ -n "$MIN_SPECULATIVE_TOKENS_PER_SWEEP" ]]; then
    analyzer_args+=(--min-speculative-tokens-per-sweep "$MIN_SPECULATIVE_TOKENS_PER_SWEEP")
  fi
  set +e
  "$ROOT_DIR/scripts/analyze_coreml_probe_log.py" "$LAUNCH_LOG" \
    "${analyzer_args[@]}" 2>&1 | tee "$ANALYZE_LOG"
  analyze_status=${PIPESTATUS[0]}
  set -e
else
  analyze_status=0
fi

echo "device console log: $LAUNCH_LOG"
if [[ "$NO_ANALYZE" != "1" ]]; then
  echo "analyzer log: $ANALYZE_LOG"
fi

if [[ "$launch_status" -ne 0 ]]; then
  echo "launch failed with status $launch_status" >&2
  exit "$launch_status"
fi
if [[ "$analyze_status" -ne 0 ]]; then
  echo "analysis failed with status $analyze_status" >&2
  exit "$analyze_status"
fi
