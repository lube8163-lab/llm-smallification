#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="/workspace/gemma12b/models/gemma-4-12b-it-qat-q4_0-unquantized"
OUT_DIR="/workspace/gemma12b/coreml-endpoints-seq4-int4"
ARCHIVE="/workspace/gemma12b/gemma4-norm-lm-head-endpoint.tar.gz"
PYTHON_BIN="${PYTHON:-python}"
SKIP_PYTHON_IMPORTS=0
ALLOW_NO_CUDA=0
MIN_FREE_GB=10

usage() {
  cat <<'EOF'
Usage:
  runpod_preflight_gemma4_coreml_endpoint.sh [options]

Options:
  --model-dir PATH       Gemma QAT-unquantized model directory.
  --out-dir PATH         Core ML endpoint .mlpackage output directory.
  --archive PATH         Transfer archive path.
  --python PATH          Python executable to inspect.
  --min-free-gb N        Minimum free space expected near the output dir.
  --skip-python-imports  Skip Python import/CUDA checks; useful for local tests.
  --allow-no-cuda        Warn instead of failing when torch cannot see CUDA.
  -h, --help             Show this help.

This is intended to run on RunPod before the endpoint conversion. It checks the
model path, writable output/archive locations, Python dependencies, CUDA
visibility, and available disk space without loading the full Gemma model.
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
    --python)
      require_value "$@"
      PYTHON_BIN="$2"
      shift 2
      ;;
    --min-free-gb)
      require_value "$@"
      MIN_FREE_GB="$2"
      shift 2
      ;;
    --skip-python-imports)
      SKIP_PYTHON_IMPORTS=1
      shift
      ;;
    --allow-no-cuda)
      ALLOW_NO_CUDA=1
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

error_count=0
warn_count=0

ok() {
  echo "OK: $*"
}

warn() {
  warn_count=$((warn_count + 1))
  echo "WARN: $*"
}

error() {
  error_count=$((error_count + 1))
  echo "ERROR: $*" >&2
}

check_dir() {
  local path="$1"
  local label="$2"
  if [[ -d "$path" ]]; then
    ok "$label exists: $path"
  else
    error "missing $label: $path"
  fi
}

echo "RunPod Gemma endpoint preflight"
echo "model dir: $MODEL_DIR"
echo "out dir: $OUT_DIR"
echo "archive: $ARCHIVE"

check_dir "$MODEL_DIR" "model dir"
if [[ -d "$MODEL_DIR" ]]; then
  if [[ -f "$MODEL_DIR/config.json" ]]; then
    ok "model config found"
  else
    error "missing model config: $MODEL_DIR/config.json"
  fi

  shopt -s nullglob
  weights=("$MODEL_DIR"/*.safetensors "$MODEL_DIR"/*.bin "$MODEL_DIR"/*.pt)
  shopt -u nullglob
  if [[ "${#weights[@]}" -gt 0 || -f "$MODEL_DIR/model.safetensors.index.json" ]]; then
    ok "model weights found"
  else
    error "missing model weight files under $MODEL_DIR"
  fi
fi

mkdir -p "$OUT_DIR" "$(dirname "$ARCHIVE")"
if [[ -w "$OUT_DIR" ]]; then
  ok "out dir writable: $OUT_DIR"
else
  error "out dir is not writable: $OUT_DIR"
fi

archive_parent="$(dirname "$ARCHIVE")"
if [[ -w "$archive_parent" ]]; then
  ok "archive parent writable: $archive_parent"
else
  error "archive parent is not writable: $archive_parent"
fi

if command -v df >/dev/null 2>&1; then
  df -h "$OUT_DIR" || true
  free_kb="$(df -Pk "$OUT_DIR" | awk 'NR==2 {print $4}')"
  if [[ "$free_kb" =~ ^[0-9]+$ ]]; then
    free_gb=$((free_kb / 1024 / 1024))
    if (( free_gb < MIN_FREE_GB )); then
      error "free space near out dir is ${free_gb} GiB, expected at least ${MIN_FREE_GB} GiB"
    else
      ok "free space near out dir is ${free_gb} GiB"
    fi
  else
    warn "could not parse df free space"
  fi
else
  warn "df command not available"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
else
  warn "nvidia-smi not found"
fi

if [[ "$SKIP_PYTHON_IMPORTS" == "1" ]]; then
  warn "skipping Python import/CUDA checks"
else
  if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    error "python executable not found: $PYTHON_BIN"
  else
    set +e
    "$PYTHON_BIN" - "$ALLOW_NO_CUDA" <<'PY'
import importlib
import sys

allow_no_cuda = sys.argv[1] == "1"
errors = 0

for name in ("numpy", "torch", "transformers", "coremltools"):
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
        print(f"OK: python import {name} {version}")
    except Exception as exc:
        errors += 1
        print(f"ERROR: python import {name} failed: {exc}", file=sys.stderr)

try:
    from transformers import AutoModelForImageTextToText  # noqa: F401
    print("OK: AutoModelForImageTextToText import")
except Exception as exc:
    errors += 1
    print(f"ERROR: AutoModelForImageTextToText import failed: {exc}", file=sys.stderr)

try:
    import torch
    cuda_ok = torch.cuda.is_available()
    print(f"OK: torch cuda available={cuda_ok}")
    if cuda_ok:
        print(f"OK: cuda device count={torch.cuda.device_count()}")
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            print(
                "OK: cuda device "
                f"{index}: {props.name}, total={props.total_memory / 1024**3:.1f} GiB"
            )
    elif allow_no_cuda:
        print("WARN: CUDA unavailable but --allow-no-cuda was passed")
    else:
        errors += 1
        print("ERROR: torch cannot see CUDA", file=sys.stderr)
except Exception as exc:
    errors += 1
    print(f"ERROR: CUDA inspection failed: {exc}", file=sys.stderr)

sys.exit(1 if errors else 0)
PY
    python_status=$?
    set -e
    if [[ "$python_status" -ne 0 ]]; then
      error "Python dependency/CUDA checks failed"
    fi
  fi
fi

if [[ "$error_count" -gt 0 ]]; then
  echo "summary: $error_count error(s), $warn_count warning(s)"
  exit 1
fi

echo "summary: 0 error(s), $warn_count warning(s)"
echo "RunPod endpoint preflight passed"
