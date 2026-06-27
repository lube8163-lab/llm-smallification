# Handoff: Gemma 4 12B Core ML iPhone probe

Use this when continuing in a fresh Codex chat.

## Project

- Repo: `/Users/tasuku/Documents/llm小型化`
- GitHub: `https://github.com/lube8163-lab/llm-smallification`
- Main app: `ios/CoreMLProbe/CoreMLProbe.xcodeproj`
- Current branch: `main`
- Large artifacts are intentionally ignored:
  - `runpod-artifacts/coreml-probes`
  - `runpod-artifacts/compiled`
  - `ios/CoreMLProbe/CoreMLProbe/Models`
  - `.derived-data/CoreMLProbe`

## Important local state

- `ios/CoreMLProbe/CoreMLProbe.xcodeproj/project.pbxproj` has a persistent local
  signing/settings diff from Xcode. Do not revert or stage it unless the user
  explicitly asks.
- The iOS app currently has the full text-only bundle locally:
  - embedding `.mlmodelc`
  - LM head `.mlmodelc`
  - 48 decoder layer `.mlmodelc` bundles
  - app/model size about `6.8G`
- RunPod is stopped. The fixed-shape 48-layer probe no longer needs an active
  pod unless reconversion or tokenizer/reference work is required.

## Proven results

- Model: `google/gemma-4-12B-it-qat-q4_0-unquantized`
- Conversion: fixed `seq=4`, no KV cache, layer-by-layer Core ML MLProgram,
  fp16 compute precision, int4 per-block weight quantization, block size 32.
- iPhone device: iPhone18,3 on iOS 26.4.2.
- CPU-only sequential load/predict/release works through all 48 decoder layers.
- Best current 48-layer log:
  - `full-stack-sequential`: peak `192.1 MB`, top logit `#253027 1.747`
  - `load-decoder-stack`: peak `252.3 MB`
  - `decoder-stack`: peak `251.8 MB`
  - `generate-one-token`: peak `252.1 MB`, next token `#253027 1.747`,
    timed steps about `66.2 sec`
  - `generate-token-loop`: peak `260.0 MB`, tokens `#253027,#253027`,
    timed steps about `150.8 sec`
  - `generate-token-loop`, `Cache = Run end`, `Tokens = 4`: peak `253.0 MB`,
    tokens `#253027,#253027,#253027,#253027`, timed steps about `243.1 sec`.
    Token 1 peaked at `253.0 MB`; tokens 2-4 peaked around `40.2 MB`.
  - `generate-token-loop`, `Cache = Run end`, `Tokens = 8`: peak `226.2 MB`,
    tokens all `#253027`, timed steps about `459.1 sec`. Token 1 took
    `67.5 sec`; tokens 2-8 averaged `54.7 sec/token` and stayed near
    `39-43 MB` peak.
- In `generate-one-token`, most time is model load/release overhead:
  about `63.7 sec` of timed steps were package loads, while all 48 decoder
  predictions totaled about `1.4 sec`.
- In the 2-token loop, timed steps were dominated by decoder package loads:
  about `120.2 sec` of `150.8 sec`; decoder predictions totaled about
  `2.8 sec`.
- Earlier 24-layer crash was traced to Core ML/E5RT runtime cache pressure. The
  app now clears `com.apple.e5rt.e5bundlecache` at run start and after each model
  release.

## Current state

- `generate-one-token` is implemented and has passed on iPhone CPU-only with
  `Layers = First 48`.
- `generate-token-loop` is implemented as the next probe. It keeps a fixed
  4-token window, appends each argmax token, and slides the window. It reuses
  embedding and LM head models across generated tokens, while decoder layers
  still use load/predict/release for memory headroom.
- Keep the fixed 4-token shape.
- Accept prompt IDs via `COREML_PROBE_INPUT_IDS` or `--input-ids=`.
- Accept generated token count via `COREML_PROBE_GENERATE_TOKENS` or
  `--tokens=`. The app clamps the UI to `1...8`.
- Accept cache-clear policy via `COREML_PROBE_CACHE_POLICY`, `--cache-policy=`,
  or the Cache picker. Values: `every-model`, `every-4-layers`,
  `every-8-layers`, `per-token`, `run-end-only`.
- Run embedding -> selected decoder layers -> last-token LM head -> argmax.
- Log `Prompt IDs` and `Next token`.
- A simple app icon exists in `Assets.xcassets/AppIcon.appiconset`.

## Recommended next step

The best current cache policy for generation is `Run end`: it keeps Core ML's
runtime cache warm during a response, then clears it after the run. The 8-token
probe is stable enough to move to text chat plumbing.

1. Add tokenizer/prompt formatting or a host-side helper that feeds known-good
   token IDs.
2. Build a chat UI around the existing fixed-window generator while keeping
   `Run end` as the generation cache policy.
3. Keep image/audio as a planned attachment surface in the UI, but do not wire
   it to inference until modality embedding/projection packages are converted.

## Multimodal notes

- The selected Gemma 4 12B checkpoint is `Gemma4UnifiedForConditionalGeneration`
  with `model_type = gemma4_unified`.
- The Hugging Face config exposes `image_token_id = 258880`,
  `audio_token_id = 258881`, `boi_token_id = 255999`,
  `boa_token_id = 256000`, `eoi_token_id = 258882`, and
  `eoa_token_index = 258883`.
- `vision_config` has `model_type = gemma4_unified_vision`,
  `patch_size = 16`, and `mm_embed_dim = 3840`; `audio_config` has
  `model_type = gemma4_unified_audio` and `audio_embed_dim = 640`.
- Treat text chat as phase 1. Image/audio support requires converting the
  modality input projection path into Core ML and feeding those embeddings into
  the same 3840-wide decoder stack. The current app only bundles text
  embedding, decoder layers, and LM head.

## Useful commands

```bash
./scripts/build_coreml_probe_ios.sh
./scripts/build_coreml_probe_ios.sh 'generic/platform=iOS'
du -sh ios/CoreMLProbe/CoreMLProbe/Models .derived-data/CoreMLProbe/Build/Products/Debug-iphoneos/CoreMLProbe.app
find ios/CoreMLProbe/CoreMLProbe/Models -maxdepth 1 -name 'gemma4_12b_layer*_decoder_seq4_mask_int4_block32.mlmodelc' -type d | wc -l
```

When committing, stage only the files changed intentionally for the task.
