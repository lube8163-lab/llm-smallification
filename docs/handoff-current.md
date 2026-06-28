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
  - `generate-token-loop`, endpoint `CPU`, decoder `CPU+GPU`, `Cache = Run end`,
    `Tokens = 8`: peak `316.2 MB`, tokens all `#253027`, token totals summed
    to about `115.9 sec`. Token 1 took `20.1 sec`; tokens 2-8 averaged about
    `13.7 sec/token`.
  - Endpoint `CPU+GPU` with decoder `CPU+GPU` passed a 1-token 48-layer run but
    peaked at `1152.0 MB` and was slower than endpoint `CPU`.
  - Endpoint `All` crashed while loading endpoint models with
    `EXC_RESOURCE (RESOURCE_TYPE_MEMORY)` at the iOS high-water limit
    (`3376 MB`), so endpoint modes that include ANE are hidden in the UI and
    blocked before endpoint model load.
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
  `--tokens=`. The app clamps the UI to `1...32`, but still defaults to `2`.
- The app now has `Chat` and `Probe` tabs. Chat wraps `generate-token-loop`
  and keeps `Cache = Run end` as the default.
- Endpoint models (embedding and LM head) and decoder layers can use separate
  compute-unit settings:
  - `COREML_PROBE_ENDPOINT_COMPUTE` or `--endpoint-compute=`
  - `COREML_PROBE_DECODER_COMPUTE` or `--decoder-compute=`
  - `COREML_PROBE_COMPUTE` or `--compute=` remains the shared fallback.
- Chat now defaults to endpoint `CPU` and decoder `CPU+GPU`, which is the
  fastest stable measured split so far. Probe still starts conservatively unless
  launch settings override it.
- Accept cache-clear policy via `COREML_PROBE_CACHE_POLICY`, `--cache-policy=`,
  or the Cache picker. Values: `every-model`, `every-4-layers`,
  `every-8-layers`, `per-token`, `run-end-only`.
- Run embedding -> selected decoder layers -> last-token LM head -> argmax.
- Log `Prompt IDs`, generated tokens, per-token total time, and peak memory.
- Log `Top logits token n` and `Repeated input window n` so repeated-token
  collapse is visible in the device console.
- `scripts/gemma4_token_helper.py` is a host-side helper for prompt-to-token
  window and generated-ID decode while the app still lacks an on-device
  tokenizer.
- A simple app icon exists in `Assets.xcassets/AppIcon.appiconset`.
- Chat composer text is not tokenized on device yet. It now accepts pasted
  `input_ids_last4=...`, `input_ids=...`, four `#123` IDs, or four raw IDs from
  the message body; otherwise it uses the `Token window` field and labels the
  message detail as `message text is not tokenized yet`.

## Recommended next step

The best current cache policy for generation is `Run end`, and the best measured
compute split is endpoint `CPU` with decoder `CPU+GPU`. That combination has
completed `Tokens = 16` through all 48 layers without crashing.

1. Next, test the same split with a tokenizer-derived non-repeated window at
   `Tokens = 16`, then try `32` if memory stays near the `300-350 MB` range.
2. Use `scripts/gemma4_token_helper.py --prompt ...` to feed better 4-token
   windows and `--decode-ids ...` to inspect generated IDs; the current default
   window tends to collapse to repeated `#253027`.
   - Latest 16-token iPhone test showed run 1 sliding from `2,123,4567,106` to
     `253027,253027,253027,253027` by token 6, then run 2 starting from that
     repeated window. At that point top logits are identical and argmax remains
     `#253027`, so natural-language composer text cannot affect output until a
     real token window is provided.
3. If probing more accelerator behavior, keep endpoint `CPU` and test decoder
   `All` from small layer counts upward (`First 1`, `8`, `16`, `32`, `48`).
4. Keep image/audio as a planned attachment surface in the UI, but do not wire
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
