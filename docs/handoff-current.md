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
- RunPod is stopped. The next quality fix requires reconversion of the endpoint
  LM head package to include final RMSNorm and final logit softcap.

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
- The app now prefers `gemma4_12b_norm_lm_head_1tok_int4_block32.mlmodelc`
  for LM head inference. This package should include language-model final
  RMSNorm, tied lm_head, and Gemma final logit softcap. If it is absent, the app
  falls back to the older linear-only `gemma4_12b_lm_head_1tok_int4_block32`
  and logs `LM head fallback`; with the preferred endpoint present it logs
  `LM head target`.
- `scripts/runpod_convert_gemma4_coreml_endpoints.py` converts the fixed-shape
  embedding package and the corrected norm+lm_head endpoint package on RunPod.
  After compiling the `.mlpackage` outputs locally, `prepare_ios_coreml_probe_assets.sh`
  copies the norm+lm_head bundle when present and removes stale legacy/new LM
  head bundles from the iOS target before copying.
- `scripts/runpod_refresh_gemma4_coreml_endpoint.sh` is the preferred RunPod
  entrypoint for the endpoint quality fix. It converts the corrected norm+lm_head
  endpoint, writes SHA256 sums, and creates a transfer tarball.
- `scripts/package_runpod_coreml_endpoint.py` can run on RunPod after endpoint
  conversion to verify the expected `.mlpackage`, write SHA256 sums, and
  optionally create a transfer `.tar.gz`.
- For the norm+lm_head-only update, use
  `prepare_ios_coreml_probe_assets.sh --endpoints-only <compiled-endpoint-dir>`;
  this replaces endpoint bundles without requiring or touching the existing
  decoder layer bundles in `ios/CoreMLProbe/CoreMLProbe/Models`.
- `scripts/verify_coreml_probe_assets.py` checks the staged iOS model bundles.
  Use `--allow-legacy-lm-head` for the current known legacy state, and use
  `--require-norm-lm-head --fail-on-legacy` after the endpoint refresh.
- `scripts/refresh_coreml_probe_endpoint.sh <endpoint-mlpackage-dir-or-tar.gz>` runs the
  local endpoint refresh flow: compile the norm+lm_head package, copy only
  endpoint bundles, verify strict asset state, then build for generic iOS.
- `scripts/gemma4_token_helper.py` is a host-side helper for prompt-to-token
  window and generated-ID decode while the app still lacks an on-device
  tokenizer. It depends on `transformers sentencepiece jinja2` and defaults to
  raw prompt tokenization, because chat-template last4 often collapses to the
  assistant-prefix tokens instead of prompt content.
- `scripts/analyze_coreml_probe_log.py` summarizes pasted/saved Xcode logs and
  can enforce the post-refresh expectations: preferred norm+lm_head endpoint,
  no fallback, expected layer count, generated token count, repeat detection,
  and peak-memory ceiling.
- A simple app icon exists in `Assets.xcassets/AppIcon.appiconset`.
- Chat composer text is not tokenized on device yet. It now accepts pasted
  `input_ids_last4=...`, `input_ids=...`, four `#123` IDs, or four raw IDs from
  the message body; otherwise it uses the `Token window` field and labels the
  message detail as `message text is not tokenized yet`.

## Recommended next step

The best current cache policy for generation is `Run end`, and the best measured
compute split is endpoint `CPU` with decoder `CPU+GPU`. That combination has
completed tokenizer-derived `Tokens = 8` and earlier `Tokens = 16` runs through
all 48 layers without crashing.

1. Reconnect RunPod or another CUDA host with the Gemma weights and run:
   `./scripts/runpod_refresh_gemma4_coreml_endpoint.sh`.
   Then run `./scripts/refresh_coreml_probe_endpoint.sh <endpoint-mlpackage-dir-or-tar.gz>`
   on the Mac to compile, copy, verify, and build.
2. Verify the device log loads
   `gemma4_12b_norm_lm_head_1tok_int4_block32` with no `LM head fallback`.
3. Save the Xcode console output and run
   `./scripts/analyze_coreml_probe_log.py <log> --require-norm-lm-head --fail-on-repeat --expect-layers 48 --min-generated-tokens 8 --max-peak-mb 350`.
4. Test endpoint `CPU`, decoder `CPU+GPU`, `Layers = First 48`,
   `Cache = Run end`, tokenizer-derived `Tokens = 8` first, then `16` if peak
   memory stays near the `300-350 MB` range.
5. Use `scripts/gemma4_token_helper.py --prompt ...` to feed better 4-token
   windows and `--decode-ids ...` to inspect generated IDs; the current default
   window tends to collapse to repeated `#253027`.
   - Latest 16-token iPhone test showed run 1 sliding from `2,123,4567,106` to
     `253027,253027,253027,253027` by token 6, then run 2 starting from that
     repeated window. At that point top logits are identical and argmax remains
     `#253027`, so natural-language composer text cannot affect output until a
     real token window is provided.
   - A later tokenizer-derived test with `237234,7604,31600,3335` confirmed the
     app used the pasted window, but it still collapsed to `#253027` by token 5.
     Inspection of the shipped LM head MIL showed it was linear-only, so the
     next fix is the norm+lm_head endpoint package above.
6. If probing more accelerator behavior, keep endpoint `CPU` and test decoder
   `All` from small layer counts upward (`First 1`, `8`, `16`, `32`, `48`).
7. Keep image/audio as a planned attachment surface in the UI, but do not wire
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
