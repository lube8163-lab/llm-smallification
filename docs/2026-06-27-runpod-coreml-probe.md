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

## iPhone CPU-only sequential smoke

Device log supplied from iPhone18,3 on iOS 26.4.2. The first implementation that
held embedding, decoder, and lm head `MLModel` instances at the same time was
terminated by iOS with code 9 for excessive memory use. After switching the
probe to CPU-only sequential load/predict/release modes, all single-layer modes
completed.

| Mode | Peak footprint in log | Notable timing |
| --- | ---: | ---: |
| `load-embedding` | 26.2 MB | initial load 4.4219 sec |
| `load-decoder` | 185.3 MB | initial load 0.6584 sec |
| `load-lm-head` | 30.6 MB | initial load 4.0908 sec |
| `load-all-sequential` | 26.5 MB | cached loads under 0.02 sec each |
| `embedding-only` | 26.8 MB | predict 0.0109 sec |
| `decoder-only` | 27.0 MB | predict 0.6092 sec |
| `lm-head-only` | 31.7 MB | predict 0.6962 sec |
| `full-sequential` | 31.8 MB | decoder 0.5693 sec, lm head 1.2053 sec |

This strongly suggests that the earlier crash was caused by simultaneous model
retention and/or accelerator/debugger runtime overhead rather than by the
single-layer CPU path itself. The next useful probe is multiple decoder layer
packages discovered and executed one at a time.

## iPhone CPU-only 8-layer stack smoke

After copying and compiling decoder layers 0-7 locally, the iPhone CPU-only
stack probes completed with the sequential loader.

| Mode | Result | Peak footprint in log | Notable timing |
| --- | --- | ---: | ---: |
| `load-decoder-stack` | loaded 8 decoder layers | 187.0 MB | initial layer loads ranged 0.78-5.09 sec |
| `decoder-stack` | decoded `[1,4,3840]` | 29.1 MB | 8 decoder predictions took about 7.0 sec total |
| `full-stack-sequential` | embedding, 8 layers, lm head completed | 34.8 MB | decoder predictions took about 7.6 sec total; lm head 0.2567 sec |

There was no obvious monotonic memory growth across the 8 decoder layers. The
full-stack output top logit was `#121220 6.859`. This is still a fixed-shape
seq=4, cache-free probe, but the sequential Core ML packaging approach looks
healthy enough to scale the layer count in steps.

## 16-layer staging for next iPhone smoke

After redeploying RunPod with the preserved network volume attached, the remote
artifact store was still intact:

- `/workspace/gemma12b/coreml-layers-seq4-int4`
- `48` layer packages
- about `6.1G`

Decoder layers 08-15 were copied to the Mac, compiled with
`compile_coreml_probe_packages.sh`, and copied into the iOS probe with
`prepare_ios_coreml_probe_assets.sh`. The probe app now contains embedding,
LM head, and the first 16 decoder layers.

| Artifact | Size / count |
| --- | ---: |
| `ios/CoreMLProbe/CoreMLProbe/Models` | about `3.0G` |
| `Debug-iphonesimulator/CoreMLProbe.app` | about `3.0G` |
| `Debug-iphoneos/CoreMLProbe.app` | about `3.0G` |
| Decoder `.mlmodelc` bundles in app | `16` |

Both simulator and generic iOS Debug builds completed successfully. The next
unknown is device runtime memory with `Layers = First 16`, not packaging or
signing.

## iPhone CPU-only 16-layer stack smoke

Device log supplied from iPhone18,3 on iOS 26.4.2. The 16-layer CPU-only
stack probes completed with no memory-pressure termination.

| Mode | Result | Peak footprint in log | Notable timing |
| --- | --- | ---: | ---: |
| `load-decoder-stack` | loaded 16 decoder layers | 206.7 MB | layer loads ranged roughly 0.79-7.66 sec |
| `decoder-stack` | decoded `[1,4,3840]` | 30.7 MB | most layer predictions were about 0.46-1.03 sec |
| `full-stack-sequential` | embedding, 16 layers, lm head completed | 34.3 MB | lm head predict 10.5347 sec |

The runtime footprint still stayed essentially flat during sequential decoder
execution. The largest observed resident footprint was in the explicit load
probe, not the full sequential path. This supports continuing the same staged
test plan at higher layer counts.

## 32-layer staging for next iPhone smoke

While the RunPod instance was still active, decoder layers 16-31 were copied to
the Mac and compiled. The iOS probe now contains embedding, LM head, and the
first 32 decoder layers, so the UI can test both `First 24` and `First 32`
without another app rebuild.

| Artifact | Size / count |
| --- | ---: |
| `ios/CoreMLProbe/CoreMLProbe/Models` | about `4.9G` |
| `Debug-iphonesimulator/CoreMLProbe.app` | about `4.9G` |
| `Debug-iphoneos/CoreMLProbe.app` | about `4.9G` |
| Decoder `.mlmodelc` bundles in app | `32` |

Both simulator and generic iOS Debug builds completed successfully for this
32-layer bundle. The next risk is install time/space and device runtime
behavior, not local packaging.

## Current limitations

- This is a fixed seq=4, cache-free layer conversion. It proves operator and
  packaging feasibility, not a full autoregressive runtime.
- Linux can save MLPackages, but cannot execute or fully validate Apple runtime
  behavior.
- Only the first 32 decoder packages were copied to the Mac and compiled. The
  full 48-layer set remains on RunPod.
- Compressed package size is not the same as peak resident memory on iPhone.
  The iPhone test must measure load/predict/release behavior on device.
- No image/audio path was converted yet. This was text-only.

## Next step

Run the CPU-only iPhone probe with `Layers = First 24` first, then repeat with
`Layers = First 32` only if 24 layers pass:

1. `load-decoder-stack`
2. `decoder-stack`
3. `full-stack-sequential`

If all three pass without memory pressure, copy/compile the next block of
decoder layers and repeat the same stepwise test.
