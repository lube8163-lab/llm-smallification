#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_CLI="$ROOT_DIR/external/llama.cpp/build/bin/llama-cli"
MODEL="$ROOT_DIR/models/google-gemma-4-12b-qat-q4_0/gemma-4-12b-it-qat-q4_0.gguf"
LOG_DIR="$ROOT_DIR/logs"

MODE="${1:-cpu}"
CTX_SIZE="${CTX_SIZE:-512}"
N_PREDICT="${N_PREDICT:-16}"
THREADS="${THREADS:-4}"
BATCH_SIZE="${BATCH_SIZE:-128}"
UBATCH_SIZE="${UBATCH_SIZE:-64}"
CACHE_TYPE_K="${CACHE_TYPE_K:-q4_0}"
CACHE_TYPE_V="${CACHE_TYPE_V:-q4_0}"
PROMPT="${PROMPT:-Say hello in one short sentence.}"
REASONING="${REASONING:-off}"

mkdir -p "$LOG_DIR"

if [[ ! -x "$LLAMA_CLI" ]]; then
  echo "llama-cli is missing. Run scripts/setup_llama_cpp.sh first." >&2
  exit 1
fi

if [[ ! -f "$MODEL" ]]; then
  echo "Model is missing. Run scripts/download_gemma4_12b_qat_gguf.sh first." >&2
  exit 1
fi

common_args=(
  --model "$MODEL"
  --ctx-size "$CTX_SIZE"
  --predict "$N_PREDICT"
  --threads "$THREADS"
  --batch-size "$BATCH_SIZE"
  --ubatch-size "$UBATCH_SIZE"
  --cache-type-k "$CACHE_TYPE_K"
  --cache-type-v "$CACHE_TYPE_V"
  --mmap
  --prompt "$PROMPT"
  --single-turn
  --reasoning "$REASONING"
  --simple-io
  --perf
)

case "$MODE" in
  cpu)
    mode_args=(--device none --gpu-layers 0 --no-kv-offload)
    ;;
  metal-low)
    mode_args=(--device MTL0 --gpu-layers "${GPU_LAYERS:-4}" --kv-offload)
    ;;
  metal-low-cpukv)
    mode_args=(--device MTL0 --gpu-layers "${GPU_LAYERS:-1}" --no-kv-offload --no-op-offload)
    ;;
  metal-fit)
    mode_args=(--device MTL0 --gpu-layers auto --fit on --fit-target "${FIT_TARGET_MIB:-1536}" --fit-ctx "$CTX_SIZE" --kv-offload)
    ;;
  *)
    echo "Usage: $0 {cpu|metal-low|metal-low-cpukv|metal-fit}" >&2
    exit 1
    ;;
esac

timestamp="$(date +%Y%m%d-%H%M%S)"
log_file="$LOG_DIR/${timestamp}-gemma4-12b-${MODE}-ctx${CTX_SIZE}-n${N_PREDICT}.log"

{
  echo "mode=$MODE"
  echo "ctx_size=$CTX_SIZE"
  echo "n_predict=$N_PREDICT"
  echo "threads=$THREADS"
  echo "batch_size=$BATCH_SIZE"
  echo "ubatch_size=$UBATCH_SIZE"
  echo "cache_type_k=$CACHE_TYPE_K"
  echo "cache_type_v=$CACHE_TYPE_V"
  echo "reasoning=$REASONING"
  echo "prompt=$PROMPT"
  echo "started_at=$(date '+%Y-%m-%dT%H:%M:%S%z')"
  echo
  echo "command:"
  printf '%q ' /usr/bin/time -l "$LLAMA_CLI" "${mode_args[@]}" "${common_args[@]}"
  echo
  echo
} | tee "$log_file"

set +e
/usr/bin/time -l "$LLAMA_CLI" "${mode_args[@]}" "${common_args[@]}" 2>&1 | tee -a "$log_file"
status="${PIPESTATUS[0]}"
set -e

{
  echo
  echo "finished_at=$(date '+%Y-%m-%dT%H:%M:%S%z')"
  echo "exit_status=$status"
} | tee -a "$log_file"

exit "$status"
