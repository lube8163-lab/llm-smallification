# Gemma 4 12B on 8 GB iPhone: Experiment Plan

Goal: prioritize the technical challenge of getting a 12B-class model to load and emit at least one token on an 8 GB iPhone, then use the failure modes to design a Core AI/Core ML split runner.

## Stage 1: llama.cpp/GGUF Baseline

Target model:

- `google/gemma-4-12B-it-qat-q4_0-gguf`
- Main GGUF: `gemma-4-12b-it-qat-q4_0.gguf`
- Size: `6,975,877,728` bytes

Initial probes:

- CPU-only, `ctx=512`, quantized KV cache.
- Metal with a very small layer offload.
- Metal with llama.cpp `--fit` enabled.

The baseline is not the final architecture. It measures what fits, what swaps, and how KV/cache choices behave before investing in Core AI/Core ML conversion.

## Stage 2: Core AI/Core ML Direction

If Stage 1 confirms the model can at least initialize under tight settings, build a split runner:

- tokenizer/embed
- transformer block groups
- final norm/lm head
- external or stateful KV cache

The first success criterion is one generated token, not usability.

## RunPod Boundary

This local Mac is Apple M2 with 8 GB RAM. It can build the app, run llama.cpp probes, and inspect GGUF metadata, but full 12B conversion or calibration should move to RunPod. Prefer 80 GB GPU memory for comfortable conversion work.

