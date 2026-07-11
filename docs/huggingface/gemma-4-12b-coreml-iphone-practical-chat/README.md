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
- a 32-token audio embedder for 16 kHz PCM input.

The prefill and decode packages for the same decoder layer intentionally carry
the same quantized weights. Hugging Face Xet can deduplicate the shared chunks.

## Directory layout

```text
models/
  endpoints/
  kv/
    prefill/
    decode/
  multimodal/
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

The script downloads this repository, verifies `SHA256SUMS`, compiles each
package, and stages the resulting `.mlmodelc` bundles under the ignored iOS
`Models` directory. Building the app still requires Xcode, an Apple signing
team, and a physical iPhone with enough free storage.

## Validated configuration

| Item | Value |
|---|---|
| Device | iPhone 14, A15 Bionic, 6 GB RAM |
| OS | iOS 26.5 |
| Text window | 320-token prefill |
| KV cache | 512 slots, up to 192 generated tokens in the app |
| Decoder quantization | 4-bit palettization, per-grouped-channel group 16 |
| Endpoint quantization | int4-block embeddings, pal4 norm/lm_head |
| Compute plan | 40 sliding-attention layers on ANE; 8 full-attention decode layers on CPU+GPU |
| Observed decode speed | approximately 0.2-0.3 token/s on iPhone 14 |

The matching application has produced meaningful Japanese text, image
descriptions, and a short answer to an English spoken question entirely
offline. Device behavior depends strongly on iPhone generation, available
memory, iOS/Core ML version, and first-run ANE compilation state.

## Provenance

- Upstream model: `google/gemma-4-12B-it-qat-q4_0-unquantized`
- Upstream weight-file revision:
  `58540658b6c08edab2ddc1fbde7f28cc9987ced3`
- Conversion/application repository:
  `https://github.com/lube8163-lab/llm-smallification`
- Conversion code snapshot documented for this release:
  `dd024517dffd3064be87a7dd70e529b8752ac638`

The upstream repository's later commit
`a89c069a80c767b0d378c4806b2953ae9d2c711d` updates its documentation; the
published weight file remains associated with the revision listed above.

## Limitations

- These are fixed-shape research graphs tailored to the accompanying app.
- The image path uses an application-level workaround for high-RMS special
  token embeddings; consult the code and article before adapting it.
- No server-side or cloud inference endpoint is provided.
- The release does not include the optional MTP speculative-decoding drafter.
- Generated content inherits the normal limitations and risks of the upstream
  model. Validate outputs for your own use case.

## License and attribution

The converted model artifacts are derived from Gemma 4 and are distributed
under the Apache License 2.0. See `LICENSE` and `NOTICE.md`. The files have been
modified through graph decomposition, fixed-shape tracing, precision changes,
and weight compression for Core ML. Apple, Core ML, iPhone, and Xcode are
trademarks of Apple Inc.; this project is not affiliated with Apple or Google.
