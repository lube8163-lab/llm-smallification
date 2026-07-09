#!/usr/bin/env bash
# Build CoreMLProbe once, install once, then run a battery of automation kinds
# on the connected iPhone back-to-back and emit one consolidated summary.
#
# Intended loop while optimizing: make code changes -> run this -> read
# artifacts/coreml-bench/<timestamp>/summary.md to see where time/memory went.
#
# Each kind reuses run_coreml_probe_device_automation.sh (build/install/launch
# via `xcrun devicectl --console` with COREML_PROBE_AUTO_EXIT=1). The first kind
# builds + installs; the rest skip both. Analyzer failures do NOT abort the
# suite — this is a benchmark, so every kind runs and metrics are collected
# regardless, then summarize_coreml_probe_bench.py parses the console logs.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SINGLE="$ROOT_DIR/scripts/run_coreml_probe_device_automation.sh"
SUMMARIZE="$ROOT_DIR/scripts/summarize_coreml_probe_bench.py"

DEVICE="${COREML_PROBE_DEVICE:-}"
# Default battery: prove the memory ceiling (ramp) + compare speed across
# resident levels (retention sweep runs retain 0/2/4/6 in one launch).
KINDS="${COREML_PROBE_SUITE_KINDS:-memory-ramp retention}"
SKIP_BUILD=0
SKIP_INSTALL=0
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
SUITE_DIR="${COREML_PROBE_SUITE_DIR:-$ROOT_DIR/artifacts/coreml-bench/$TIMESTAMP}"

usage() {
  cat <<'EOF'
Usage:
  bench_coreml_probe_device_suite.sh [options]

Options:
  --device ID        CoreDevice identifier/UDID/name (default: first iPhone).
  --kinds "A B C"    Space-separated automation kinds to run in order.
                     Default: "memory-ramp retention".
                     Valid: memory-ramp, retention, speed, stability,
                     chat, probe, tokens32, image, audio, multimodal.
  --suite-dir PATH   Output directory (default: artifacts/coreml-bench/<ts>).
  --skip-build       Reuse the existing Debug-iphoneos app for the first kind.
  --skip-install     Do not (re)install before the first kind.
  -h, --help         Show this help.

Environment:
  COREML_PROBE_DEVICE, COREML_PROBE_SUITE_KINDS, COREML_PROBE_SUITE_DIR.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) DEVICE="$2"; shift 2 ;;
    --kinds) KINDS="$2"; shift 2 ;;
    --suite-dir) SUITE_DIR="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --skip-install) SKIP_INSTALL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

mkdir -p "$SUITE_DIR"
echo "CoreMLProbe bench suite"
echo "kinds:  $KINDS"
echo "device: ${DEVICE:-<auto-detect>}"
echo "output: $SUITE_DIR"
echo

STATUS_FILE="$SUITE_DIR/status.txt"
: > "$STATUS_FILE"

first=1
for kind in $KINDS; do
  kind_dir="$SUITE_DIR/$kind"
  echo "=== running kind: $kind -> $kind_dir ==="
  args=(--kind "$kind" --log-dir "$kind_dir")
  [[ -n "$DEVICE" ]] && args+=(--device "$DEVICE")

  # Resident/ramp runs peak in the multi-GB range; lift the analyzer ceiling so
  # a legitimately high peak is not reported as a failure.
  case "$kind" in
    retention|retain|retention-sweep|retain-sweep)
      args+=(--max-peak-mb 8192)
      ;;
  esac

  if [[ $first -eq 1 ]]; then
    [[ $SKIP_BUILD -eq 1 ]] && args+=(--skip-build)
    [[ $SKIP_INSTALL -eq 1 ]] && args+=(--skip-install)
  else
    # App is already built and installed by the first kind.
    args+=(--skip-build --skip-install)
  fi
  first=0

  set +e
  "$SINGLE" "${args[@]}"
  status=$?
  set -e
  echo "$kind exit=$status" | tee -a "$STATUS_FILE"
  echo
done

SUMMARY_MD="$SUITE_DIR/summary.md"
echo "=== summarizing ==="
python3 "$SUMMARIZE" "$SUITE_DIR" | tee "$SUMMARY_MD"

echo
echo "per-kind exit status:"
cat "$STATUS_FILE"
echo
echo "summary:      $SUMMARY_MD"
echo "console logs: $SUITE_DIR/<kind>/device-console.log"
