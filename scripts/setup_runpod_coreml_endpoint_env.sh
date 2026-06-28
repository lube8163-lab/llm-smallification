#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
DRY_RUN=0
UPGRADE=0
INSTALL_TORCH=0
TORCH_INDEX_URL=""

usage() {
  cat <<'EOF'
Usage:
  setup_runpod_coreml_endpoint_env.sh [options]

Options:
  --python PATH         Python executable to use.
  --upgrade             Pass --upgrade to pip installs.
  --install-torch       Install torch as well as the surrounding dependencies.
  --torch-index-url URL Optional PyTorch wheel index URL, e.g. a CUDA wheel index.
  --dry-run             Print commands without executing them.
  -h, --help            Show this help.

By default this avoids reinstalling torch, because RunPod PyTorch images usually
ship a CUDA-enabled torch build already. It installs the conversion-side Python
packages used by runpod_convert_gemma4_coreml_endpoints.py and the preflight.
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
    --python)
      require_value "$@"
      PYTHON_BIN="$2"
      shift 2
      ;;
    --upgrade)
      UPGRADE=1
      shift
      ;;
    --install-torch)
      INSTALL_TORCH=1
      shift
      ;;
    --torch-index-url)
      require_value "$@"
      TORCH_INDEX_URL="$2"
      shift 2
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

pip_args=("$PYTHON_BIN" -m pip install)
if [[ "$UPGRADE" == "1" ]]; then
  pip_args+=(--upgrade)
fi

if [[ "$INSTALL_TORCH" == "1" ]]; then
  torch_args=("${pip_args[@]}")
  if [[ -n "$TORCH_INDEX_URL" ]]; then
    torch_args+=(--index-url "$TORCH_INDEX_URL")
  fi
  torch_args+=(torch)
  run_cmd "${torch_args[@]}"
fi

run_cmd "${pip_args[@]}" \
  numpy \
  coremltools \
  transformers \
  accelerate \
  safetensors \
  sentencepiece \
  protobuf

run_cmd "$PYTHON_BIN" - <<'PY'
import importlib

for name in (
    "numpy",
    "torch",
    "transformers",
    "accelerate",
    "safetensors",
    "coremltools",
):
    module = importlib.import_module(name)
    version = getattr(module, "__version__", "unknown")
    print(f"{name} {version}")

from transformers import AutoModelForImageTextToText  # noqa: F401

import torch

print(f"torch cuda available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda device count={torch.cuda.device_count()}")
PY

if [[ "$DRY_RUN" == "1" ]]; then
  echo "RunPod Core ML endpoint Python environment setup dry run complete"
else
  echo "RunPod Core ML endpoint Python environment setup complete"
fi
