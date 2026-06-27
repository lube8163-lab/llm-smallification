# 2026-06-27 llama.cpp Baseline

Host:

- Apple M2
- 8 GB RAM
- macOS 26.5.1
- Xcode 26.5
- llama.cpp `050ee92`

Model:

- Repo: `google/gemma-4-12B-it-qat-q4_0-gguf`
- File: `models/google-gemma-4-12b-qat-q4_0/gemma-4-12b-it-qat-q4_0.gguf`
- Size: `6,975,877,728` bytes

## Results

| Mode | Key settings | Result |
| --- | --- | --- |
| `cpu` | `ctx=512`, `n=8`, q4 KV, mmap | Loaded and began generating thinking text, but was interrupted due very slow output. Observed RSS around 1.07 GB while running. |
| `cpu` | `ctx=512`, `n=4`, q4 KV, mmap, reasoning off | Loaded, no crash, but no output after about 2 minutes. Interrupted. Observed RSS around 1.06 GB. |
| `metal-low` | 4 GPU layers, GPU KV, `batch=64`, q4 KV | Failed with `kIOGPUCommandBufferCallbackErrorOutOfMemory` and `Compute error`. Max RSS `3,839,836,160`; peak footprint `5,946,861,216`. |
| `metal-low-cpukv` | 1 GPU layer, CPU KV, no op offload, `batch=16`, q4 KV | Succeeded and generated `Hello!`. Prompt `0.2 t/s`, generation `0.1 t/s`. Max RSS `2,643,312,640`; peak footprint `6,333,687,696`. |
| `metal-fit` | auto GPU layers, GPU KV, `fit-target=2048`, `batch=16`, q4 KV | Failed with `kIOGPUCommandBufferCallbackErrorOutOfMemory` and `Compute error`. Max RSS `2,628,468,736`; peak footprint `3,632,078,912`. |

## Interpretation

The official Gemma 4 12B QAT GGUF can be loaded on an 8 GB Apple Silicon machine with mmap. The first successful local generation path was not full Metal offload; it was a highly constrained hybrid:

- only 1 transformer layer on Metal
- KV cache kept off GPU
- op offload disabled
- tiny batch/ubatch
- short context
- reasoning disabled

For iPhone work, the starting point should mirror the successful path rather than relying on auto-fit:

```bash
CTX_SIZE=512 \
N_PREDICT=4 \
BATCH_SIZE=16 \
UBATCH_SIZE=16 \
CACHE_TYPE_K=q4_0 \
CACHE_TYPE_V=q4_0 \
GPU_LAYERS=1 \
./scripts/run_gemma4_12b_llama_cpp.sh metal-low-cpukv
```

## Next Step

Build an iOS probe target around llama.cpp first, because it already proves the tight memory configuration can emit a token. Use it to measure actual iPhone limits before starting Core AI/Core ML conversion. The Core AI/Core ML path should preserve the same memory lesson: tiny resident block groups, CPU-side or explicitly quantized KV, and no broad GPU/autofit assumptions.

