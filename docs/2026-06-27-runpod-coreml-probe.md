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

For later endpoint-only refreshes such as the corrected norm+lm_head package,
compile into a separate local output directory and copy only endpoint bundles:

```bash
./scripts/refresh_coreml_probe_endpoint.sh <endpoint-mlpackage-dir>
```

The wrapper runs the equivalent lower-level steps:

```bash
./scripts/compile_coreml_probe_packages.sh <endpoint-mlpackage-dir> runpod-artifacts/compiled-endpoints
./scripts/prepare_ios_coreml_probe_assets.sh --endpoints-only runpod-artifacts/compiled-endpoints
./scripts/verify_coreml_probe_assets.py --require-norm-lm-head --fail-on-legacy
```

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

## iPhone CPU-only 24-layer cache failure

Device log supplied from iPhone18,3 on iOS 26.4.2. With the 32-layer app
installed, the `First 24` run did not fail at the first decoder load boundary:

| Mode | Result | Peak footprint in log | Notable detail |
| --- | --- | ---: | --- |
| `load-decoder-stack` | loaded 24 decoder layers | 890.8 MB | E5RT reported `No space left on device` while compiling BNNS around layer 21 |
| `decoder-stack` | decoded `[1,4,3840]` | 190.3 MB | completed all 24 decoder predictions |
| `full-stack-sequential` | failed at startup | 30.8 MB before failure | E5RT failed to preallocate `com.apple.e5rt.e5bundlecache/.../bnns_program.bnnsir` |

The sharp jump near layer 21 appears to be Core ML/E5RT compile-cache pressure,
not ordinary retained `MLModel` memory. The app now clears
`com.apple.e5rt.e5bundlecache` at run start and after each model release, then
logs `Clear Core ML cache` when it removes stale runtime cache data. This makes
the probe slower, but it better matches the staged streaming experiment because
old compiled BNNS bundles should not accumulate while measuring sequential
load/predict/release behavior.

## iPhone CPU-only 24-layer cache-cleaning smoke

After deleting the old app install and reinstalling the cache-cleaning build,
`First 24` completed `full-stack-sequential` on iPhone18,3 with CPU-only
execution.

| Mode | Result | Peak footprint in log | Notable timing |
| --- | --- | ---: | ---: |
| `full-stack-sequential` | embedding, 24 layers, lm head completed | 188.1 MB | decoder predictions were about 0.03 sec after load; lm head 0.2239 sec |

`Clear Core ML cache` appeared after each model release, confirming that
`com.apple.e5rt.e5bundlecache` was removed during the run. The previous
24-layer failure was therefore most likely stale or accumulated Core ML runtime
cache pressure rather than a direct 24-layer memory ceiling.

## iPhone CPU-only 32-layer cache-cleaning smoke

Device log supplied from iPhone18,3 on iOS 26.4.2. With the cache-cleaning build
and a clean reinstall, the first 32 decoder layers completed the same CPU-only
sequential probes.

| Mode | Result | Peak footprint in log | Notable timing / output |
| --- | --- | ---: | --- |
| `full-stack-sequential` | embedding, 32 layers, lm head completed | 251.7 MB | top logit `#258882 46.938`; lm head predict 0.2224 sec |
| `load-decoder-stack` | loaded 32 decoder layers | 251.6 MB | largest observed load footprint around layer 30 |
| `decoder-stack` | decoded `[1,4,3840]` | 190.8 MB | completed all 32 decoder predictions |

The sharp 24-layer failure did not reproduce after cache clearing and reinstall.
The 32-layer result keeps the sequential package strategy viable for the full
48-layer fixed-shape probe.

## 48-layer staging for next iPhone smoke

The preserved RunPod volume was reattached to an RTX A6000 pod, and decoder
layers 32-47 were copied from
`/workspace/gemma12b/coreml-layers-seq4-int4` to the Mac. All 48 decoder layer
packages are now present locally, compiled, and copied into the iOS probe app
with the embedding and LM head bundles.

| Artifact | Size / count |
| --- | ---: |
| `runpod-artifacts/coreml-probes` | about `6.8G` |
| `runpod-artifacts/compiled` | about `6.8G` |
| `ios/CoreMLProbe/CoreMLProbe/Models` | about `6.8G` |
| `Debug-iphonesimulator/CoreMLProbe.app` | about `6.8G` |
| `Debug-iphoneos/CoreMLProbe.app` | about `6.8G` |
| Decoder `.mlmodelc` bundles in app | `48` |

Both simulator and generic iOS Debug builds completed successfully for the
48-layer bundle. The generic iOS build initially hit the usual code-signing
failure for copied resource extended attributes (`resource fork, Finder
information, or similar detritus not allowed`); clearing attributes from
`Models` resolved it, and `build_coreml_probe_ios.sh` now also clears attributes
inside each model bundle before invoking `xcodebuild`.

The app now exposes `First 40` and `First 44` layer selections in addition to
`First 48`. The recommended device order is:

1. Delete the old app install if the app container/runtime cache may be stale.
2. Run `Compute = CPU`, `Layers = First 48`, `Mode = full-stack-sequential`.
3. If `First 48` fails, bisect with `First 40`, then `First 44`.
4. If `full-stack-sequential` passes, run `load-decoder-stack` and
   `decoder-stack` for `First 48`.

## iPhone CPU-only 48-layer cache-cleaning smoke

Device log supplied from iPhone18,3 on iOS 26.4.2. With RunPod stopped and the
48-layer bundle installed locally, the full fixed-shape text stack completed on
device with CPU-only sequential load/predict/release.

| Mode | Result | Peak footprint in log | Notable timing / output |
| --- | --- | ---: | --- |
| `full-stack-sequential` | embedding, 48 layers, lm head completed | 192.1 MB | top logit `#253027 1.747` |
| `load-decoder-stack` | loaded 48 decoder layers | 252.3 MB | completed all 48 layer load/release cycles |
| `decoder-stack` | decoded `[1,4,3840]` through 48 layers | 251.8 MB | completed all 48 decoder predictions |
| `generate-one-token` | prompt IDs `2,123,4567,106`, 48 layers, last-token lm head, argmax completed | 252.1 MB | next token `#253027 1.747`; timed steps about 66.2 sec |
| `generate-token-loop` | 2 argmax tokens with sliding fixed 4-token window | 260.0 MB | tokens `#253027,#253027`; timed steps about 150.8 sec |
| `generate-token-loop`, `run-end-only` | 4 argmax tokens with sliding fixed 4-token window | 253.0 MB | tokens `#253027,#253027,#253027,#253027`; timed steps about 243.1 sec |
| `generate-token-loop`, `run-end-only` | 8 argmax tokens with sliding fixed 4-token window | 226.2 MB | all tokens `#253027`; timed steps about 459.1 sec |

This is the first end-to-end fixed-shape `seq=4` CPU-only proof that all 48
int4 decoder packages can be streamed on the target iPhone without memory
pressure termination. The result does not yet cover a real autoregressive token
loop, tokenizer integration, KV cache, or accelerator scheduling, but the
one-token argmax path confirms that the fixed-shape stack can produce an output
token on device.

The `generate-one-token` timing is dominated by package load/release overhead:
about 63.7 sec of timed steps were model loads, about 0.9 sec was cache
cleanup, and the 48 decoder predictions themselves totaled about 1.4 sec.
That makes the next optimization target package residency/caching strategy
rather than decoder matmul time.

The 2-token loop confirms the minimal repeated-generation path. Its timed
steps were about 150.8 sec total: decoder package loads were about 120.2 sec,
decoder predictions about 2.8 sec, LM head predictions about 3.6 sec, endpoint
model loads about 22.3 sec, and cache cleanup about 1.8 sec. The memory peak
remained low enough that cache-clear frequency is now worth testing directly.

Cache-policy testing then showed that `run-end-only` is the best current
generation policy. `Every 8 layers`, `Every 4 layers`, and `Per token` all
completed, but were slower than the original `every-model` baseline in the
2-token test. `Run end` completed 2 tokens in about 141.5 sec with a peak near
261 MB, then completed 4 tokens in about 243.1 sec with a peak near 253 MB. In
the 4-token run, token 1 still carried the initial cache/build cost
(`72.8 sec`, peak `253.0 MB`), while tokens 2-4 stabilized around
`52.9-55.4 sec/token` and peak memory around `40.2 MB`. Decoder package load
time after token 1 fell to roughly `0.5-0.8 sec` per full 48-layer pass; the
steady-state cost moved to decoder prediction, about `49-50 sec/token`.

The 8-token `run-end-only` run confirmed the same steady state. Timed steps
totaled about `459.1 sec`, with a peak footprint of `226.2 MB`. Token 1 took
`67.5 sec`; tokens 2-8 averaged about `54.7 sec/token` and stayed near
`39-43 MB` peak. Decoder package load after token 1 remained under about
`0.9 sec` per full 48-layer pass, while decoder prediction dominated at about
`49-54 sec/token`.

## Multimodal 12B notes

Official Gemma 4 material describes the 12B Unified model as supporting text,
image, and audio input with text output. The local Hugging Face config for the
selected checkpoint reports `architectures = ["Gemma4UnifiedForConditionalGeneration"]`
and `model_type = gemma4_unified`, with `image_token_id = 258880` and
`audio_token_id = 258881`. It also includes `vision_config` and `audio_config`;
the vision path has `patch_size = 16` and `mm_embed_dim = 3840`, while the audio
path has `audio_embed_dim = 640`.

The current Core ML bundle is text-only: token embedding, 48 decoder layer
packages, and LM head. A chat UI can expose attachment slots early, but actual
image/audio inference requires converting the modality embedding/projection
path and feeding those embeddings into the same 3840-wide decoder stack.

## Current limitations

- This is a fixed seq=4, cache-free layer conversion. It proves operator and
  packaging feasibility, not a full autoregressive runtime.
- Linux can save MLPackages, but cannot execute or fully validate Apple runtime
  behavior.
- The full 48-layer text stack has been copied to the Mac, compiled, bundled,
  and run on iPhone. RunPod is no longer needed for this fixed-shape probe.
- Compressed package size is not the same as peak resident memory on iPhone.
  The iPhone test must measure load/predict/release behavior on device.
- The app still uses fixed-shape input IDs. There is no on-device tokenizer,
  real prompt formatting, repeated autoregressive token loop, KV cache, or
  accelerator scheduling yet.
- No image/audio path was converted yet. This remains text-only.

## Next step

Use `run-end-only` as the preferred generation cache policy and move toward a
text chat surface.

1. Add tokenizer/prompt formatting or a host-side helper that feeds known-good
   token IDs.
2. Wrap the current fixed-window generator in a simple text chat UI.
3. Keep image/audio as future attachment slots until modality projection
   conversion is proven.
