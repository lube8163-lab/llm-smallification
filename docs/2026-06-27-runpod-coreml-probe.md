# 2026-06-27 RunPod Core ML probe for Gemma 4 12B

## Environment

- RunPod: A100 PCIe 80GB, pod SSH `root@69.19.136.195 -p 49911`
- Workdir: `/workspace/gemma12b`
- Python: 3.12.3
- Torch: 2.8.0+cu128
- Transformers: 5.12.1
- CoreMLTools: 9.0
- Model: `google/gemma-4-12B-it-qat-q4_0-unquantized`
- GGUF reference: `google/gemma-4-12B-it-qat-q4_0-gguf`

CoreMLTools on Linux warned that Torch 2.8 is newer than the tested 2.7 line, and
that `libcoremlpython` is unavailable. Conversion and saving still worked; actual
compile/runtime checks were done on the local Mac.

## Reference Transformers smoke

The BF16 full model loaded on A100 and generated a short response.

- Load time: 6.72 sec
- CUDA allocated: 22.277 GiB
- CUDA max allocated: 22.297 GiB
- Prompt tokens: 20
- New tokens: 3
- Generate time: 0.89 sec
- Throughput: 3.366 tok/s
- Output: `Hello!<turn|>`

## Gemma 4 text config notes

- Hidden size: 3840
- Layers: 48
- Attention heads: 16
- KV heads: 8
- Layer types: five `sliding_attention` layers followed by one `full_attention`, repeated
- Layer 46 and 47 set `store_full_length_kv=True`
- The embedding and lm head weights are tied in Transformers, but separate Core ML
  packages duplicate them unless we build a shared packaging strategy.

## Core ML component probes

All probes used MLProgram, iOS 18 minimum target, fp16 compute precision, and
CoreMLTools `linear_quantize_weights` with int4, per-block, block size 32.

| Component | Shape | FP16 package | Int4 package | Result |
| --- | --- | ---: | ---: | --- |
| Layer 0 MLP | `[1,4,3840] -> [1,4,3840]` | 0.330 GiB | not tested | Converted |
| Layer 0 attention, no mask | `[1,4,3840]` | 0.088 GiB | not tested | Converted |
| Layer 0 decoder, mask | `[1,4,3840]` | 0.418 GiB | 0.118 GiB | Converted, quantized |
| Layer 47 decoder, mask | `[1,4,3840]` | 0.451 GiB | 0.127 GiB | Converted, quantized |
| Embedding | `[1,4] -> [1,4,3840]` | 1.875 GiB | 0.527 GiB | Converted, quantized |
| LM head | `[1,1,3840] -> [1,1,262144]` | 1.875 GiB | 0.528 GiB | Converted, quantized |

## Full fixed-shape layer conversion

Script:

```bash
python scripts/runpod_convert_gemma4_coreml_layers.py \
  --layers all \
  --out-dir /workspace/gemma12b/coreml-layers-seq4-int4
```

Result:

- Converted 48 / 48 decoder layers
- Static shape: seq len 4
- Inputs per layer: `x`, `position_ids`, `attention_mask`
- Output per layer: `y`
- Quantization: int4, per-block, block size 32
- Script-reported total layer package size: 5.715 GiB
- `du -sh /workspace/gemma12b/coreml-layers-seq4-int4`: 6.1G
- Package count: 48

Adding the int4 embedding and lm head gives a binary-size estimate of about
6.77 GiB for the text-only package set:

```text
5.715 GiB decoder layers
+0.527 GiB embedding
+0.528 GiB lm_head
=6.770 GiB
```

This is close enough to an 8GB iPhone that sequential loading/releasing is not
optional. It also does not include tokenizer assets, app overhead, KV cache, or
runtime temporary buffers.

## Local Mac Core ML checks

Copied selected int4 packages to:

- `/Users/tasuku/Documents/llm小型化/runpod-artifacts/coreml-probes/`
- `/Users/tasuku/Documents/llm小型化/runpod-artifacts/compiled/`

`runpod-artifacts/` is ignored by git.

Compiled with:

```bash
xcrun coremlcompiler compile <package.mlpackage> runpod-artifacts/compiled
```

Compiled package sizes:

- Layer 0 decoder int4: 120M
- Embedding int4: 540M
- LM head int4: 541M

Runtime smoke on local Mac:

| Component | Compute units | Load | Prediction |
| --- | --- | ---: | ---: |
| Layer 0 decoder int4 | cpuOnly | 0.022 sec | 0.0085 sec avg |
| Layer 0 decoder int4 | cpuAndGPU | 0.972 sec | 0.0037 sec avg |
| Layer 0 decoder int4 | all | 0.351 sec | 0.0047 sec avg |
| Layer 0 decoder int4 | cpuAndNeuralEngine | 1.575 sec | 0.0092 sec avg |
| Embedding int4 | cpuOnly | 0.076 sec | 0.0141 sec |
| LM head int4 | cpuOnly | 0.021 sec | 4.5899 sec |
| LM head int4 | cpuAndGPU | 1.479 sec | 0.0765 sec |
| LM head int4 | all | 4.027 sec | 0.0169 sec |
| LM head int4 | cpuAndNeuralEngine | 4.658 sec | 1.0774 sec |

The LM head is a major CPU bottleneck; accelerator-backed execution is essential.

## Current limitations

- This is a fixed seq=4, cache-free layer conversion. It proves operator and
  packaging feasibility, not a full autoregressive runtime.
- Linux can save MLPackages, but cannot execute or fully validate Apple runtime
  behavior.
- Only selected packages were copied to the Mac and compiled. The full 48-layer
  set remains on RunPod.
- Compressed package size is not the same as peak resident memory on iPhone.
  The iPhone test must measure load/predict/release behavior on device.
- No image/audio path was converted yet. This was text-only.

## Next step

Build an iOS Core ML harness that loads:

1. Embedding package
2. One decoder layer package
3. LM head package

Then measure load time, prediction time, and peak memory on the actual 8GB
iPhone. After that, extend the harness to sequentially run multiple layer
packages while releasing each previous `MLModel`.
