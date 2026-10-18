---
license: apache-2.0
library_name: coreml
pipeline_tag: any-to-any
base_model: google/gemma-4-12B-it-qat-q4_0-unquantized
tags:
  - coreml
  - ios
  - iphone
  - gemma
  - multimodal
  - text-generation
---

# Gemma 4 12B Core ML for iPhone practical chat

This repository contains Core ML packages used to run a fully offline
text/image/audio chat prototype on an iPhone. The model was converted from
[`google/gemma-4-12B-it-qat-q4_0-unquantized`](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-unquantized)
and split into fixed-shape endpoint, prefill, and per-token decode graphs so it
can run within mobile memory limits.

This is an experimental research release, not a drop-in Transformers model.
The matching SwiftUI application, conversion scripts, and device notes are in
[`lube8163-lab/llm-smallification`](https://github.com/lube8163-lab/llm-smallification).

## What is included

- 48 pal4 group-16 KV-prefill layer packages with a fixed 320-token window;
- 48 pal4 group-16 KV-decode layer packages with a 512-slot cache;
- int4-block text embedding endpoints for the 320-position prefill and
  single-token decode paths;
- a pal4 group-16 norm/language-model-head endpoint;
- a 256-patch image embedder using fp32 compute where required to avoid fp16
  RMSNorm overflow;
- a 32-token audio embedder for 16 kHz PCM input;
- 48 pal4 group-16 speculative verifier packages with a fixed width of four;
- the official Gemma 4 MTP assistant converted as a mixed pal4-body/int8-head
  one-step drafter;
- the measured-useful fused prefill/verifier pair for decoder layers 00...05.

The prefill and decode packages for the same decoder layer intentionally carry
the same quantized weights. Hugging Face Xet can deduplicate the shared chunks.
The optional speculative assets preserve target-greedy output: the target
verifier, not the drafter, selects every emitted token.

## Directory layout

```text
models/
  endpoints/
  fused/
    kv/
  kv/
    prefill/
    decode/
  multimodal/
  speculative/
    drafter/
    verify/
```

`SHA256SUMS` covers every file in the published Core ML packages. The packages
are distributed as `.mlpackage` sources; compile them on macOS with
`xcrun coremlcompiler compile` rather than treating a locally generated
`.mlmodelc` bundle as the portable release artifact.

## Reproduce the iPhone application setup

Clone the code repository and run:

```bash
./scripts/download_hf_gemma4_coreml_models.sh
```

This downloads only the practical-chat base set by default. To add the
speculative verifier, MTP drafter, and selected six-layer fusion:

```bash
HF_MODEL_PROFILE=speculative \
  ./scripts/download_hf_gemma4_coreml_models.sh
```

The script verifies `SHA256SUMS`, compiles the selected packages, and stages
the resulting `.mlmodelc` bundles under the ignored iOS `Models` directory.
Building the app still requires Xcode, an Apple signing team, and a physical
iPhone with enough free storage.

## Validated configuration

| Item | Value |
|---|---|
| Devices | iPhone 14 (A15, 6 GB) and iPhone 17 / iPhone18,3 |
| OS | iOS 26.5 |
| Text window | 320-token prefill |
| KV cache | 512 slots, up to 192 generated tokens in the app |
| Decoder quantization | 4-bit palettization, per-grouped-channel group 16 |
| Endpoint quantization | int4-block embeddings, pal4 norm/lm_head |
| Speculative width | 4 rows: current token plus up to 3 drafts |
| Compute plan | Device-specific ANE / CPU+GPU routing; selected fused group on CPU+GPU |
| iPhone 14 high-acceptance probe | 4.93 to 2.06 seconds/token (2.40x) |
| iPhone 17, selected fused run | 24 tokens / 37.56 seconds = 0.639 token/s |

The matching application has produced meaningful Japanese text, image
descriptions, and a short answer to an English spoken question entirely
offline. Device behavior depends strongly on iPhone generation, available
memory, prompt-dependent draft acceptance, iOS/Core ML version, and first-run
compilation state. The 2.40x result is a selected high-acceptance 8-token
probe, not a universal throughput claim; a 24-token low-acceptance prompt
improved only from 4.20 to 4.12 seconds/token.

## Provenance

- Upstream model: `google/gemma-4-12B-it-qat-q4_0-unquantized`
- MTP assistant: `google/gemma-4-12B-it-assistant`
- Upstream weight-file revision:
  `58540658b6c08edab2ddc1fbde7f28cc9987ced3`
- Conversion/application repository:
  `https://github.com/lube8163-lab/llm-smallification`
- The release model card is maintained beside the conversion code so the
  exact publication commit can be audited from the repository history.

The upstream repository's later commit
`a89c069a80c767b0d378c4806b2953ae9d2c711d` updates its documentation; the
published weight file remains associated with the revision listed above.

## Limitations

- These are fixed-shape research graphs tailored to the accompanying app.
- The image path uses an application-level workaround for high-RMS special
  token embeddings; consult the code and article before adapting it.
- No server-side or cloud inference endpoint is provided.
- Speculative speedup is prompt dependent and can approach zero when the
  drafter's candidates are rejected.
- Only fused layers 00...05 are published. Adding layers 06...11 made the
  measured 24-token iPhone 17 run 5.4% slower and increased peak memory.
- The fused group is routed to CPU+GPU in the validated configuration. Its
  first ANE plan build exceeded iOS's per-process disk-write budget.
- Generated content inherits the normal limitations and risks of the upstream
  model. Validate outputs for your own use case.

## License and attribution

The converted model artifacts are derived from Gemma 4 and are distributed
under the Apache License 2.0. See `LICENSE` and `NOTICE.md`. The files have been
modified through graph decomposition, fixed-shape tracing, precision changes,
and weight compression for Core ML. Apple, Core ML, iPhone, and Xcode are
trademarks of Apple Inc.; this project is not affiliated with Apple or Google.
