# Handoff: Gemma 4 12B Core ML iPhone probe

Use this when continuing in a fresh Codex chat.

## Project

- Repo: `/Users/tasuku/Documents/llm小型化`
- GitHub: `https://github.com/lube8163-lab/llm-smallification`
- Main app: `ios/CoreMLProbe/CoreMLProbe.xcodeproj`
- Current branch: `main`
- New multimodal smoke-test direction: `docs/multimodal-iphone-plan.md`
- Large artifacts are intentionally ignored:
  - `runpod-artifacts/coreml-probes`
  - `runpod-artifacts/compiled`
  - `ios/CoreMLProbe/CoreMLProbe/Models`
  - `.derived-data/CoreMLProbe`

## Important local state

- `ios/CoreMLProbe/CoreMLProbe.xcodeproj/project.pbxproj` has a persistent local
  signing/settings diff from Xcode. Do not revert or stage it unless the user
  explicitly asks.
- The iOS app currently has the full text bundle locally plus the first real
  modality smoke frontends:
  - embedding `.mlmodelc`
  - LM head `.mlmodelc`
  - 48 decoder layers covered by 12 compiled 4-layer decoder chunk bundles
  - 32-patch image embedder `.mlmodelc`
  - 32-token audio embedder `.mlmodelc`
  - app/model size about `6.8G`
- RunPod is not required for the current app-side stability tests. The iOS app
  now has the corrected norm+lm_head endpoint locally, and recent device logs
  confirmed it loads without LM head fallback.
- Latest RunPod conversion on 2026-07-04 used an A100 SXM 80GB pod mounted with
  the user's `gemma12b~~` storage. `/workspace/gemma12b` now contains the venv,
  downloaded 23GB model, conversion logs, and the generated image/audio embedder
  packages. The pod is not needed for local app-side loops unless converting a
  fuller processor endpoint.

## Proven results

- Model: `google/gemma-4-12B-it-qat-q4_0-unquantized`
- Current conversion path: fixed `seq=64`, no KV cache, layer-by-layer Core ML MLProgram,
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
  - `generate-token-loop`, endpoint `CPU`, decoder `All`, `Cache = Run end`,
    `Tokens = 8`, corrected norm+lm_head endpoint, bundled tokenizer, Seq64:
    peak `666.6 MB`; token 1 took about `31.0 sec`; tokens 2-8 averaged about
    `12.8 sec/token`; generated
    `#98275,#85896,#237354,#107,#237328,#226391,#120431,#95616`.
  - `Run Stability Sweep`, endpoint `CPU`, decoder `All`, `Cache = Run end`,
    `Layers = First 48`, Retain 0, Seq64:
    - `Tokens = 8`: finished, warm average `12.92 sec/token`, peak `611.3 MB`,
      generated `#85141,#236924,#52119,#12553,#237221,#66447,#237669,#237230`.
    - `Tokens = 16`: finished, warm average `13.19 sec/token`, peak `605.2 MB`,
      generated `#85141,#236924,#52119,#12553,#237221,#66447,#237669,#237230,#237214,#3652,#236924,#108,#99844,#70622,#237000,#106241`.
    - The run used the preferred norm+lm_head endpoint, had no LM head fallback,
      selected all 48 decoder layers, and showed no repeated-window collapse.
  - Latest stability sweep recheck with the same endpoint `CPU`, decoder `All`,
    `Cache = Run end`, `Layers = First 48`, Retain 0, Seq64:
    - `Tokens = 8`: finished, warm average `11.57 sec/token`, peak `614.7 MB`,
      same first 8 generated IDs as above.
    - `Tokens = 16`: finished, warm average `12.53 sec/token`, peak `611.4 MB`,
      generated the same 16-token sequence as above.
    - Timing breakdown showed warm tokens are load-bound: roughly
      `8.2-8.5 sec/token` in decoder bundle loads, `0.9-1.2 sec/token` in
      decoder prediction, and `2.2-2.6 sec/token` in the LM head.
  - `Run Retain Sweep`, endpoint `CPU`, decoder `All`, `Cache = Run end`,
    `Layers = First 48`, Seq64:
    - Retain 0: finished, warm average `12.18 sec/token`, peak `609.1 MB`,
      generated `#85141,#236924,#52119,#12553,#237221,#66447,#237669,#237230`.
    - Retain 1: finished, warm average `12.32 sec/token`, peak `1101.9 MB`,
      generated the same 8-token sequence.
    - Retain 1 did not improve speed and raised memory sharply, so keep Retain
      0 as the default.
  - Automated real-device smoke via `scripts/run_coreml_probe_device_automation.sh`:
    - `--kind probe --tokens 1`: installed/launched with `devicectl --console`,
      auto-ran, auto-exited with code 0, analyzer passed, peak `587.7 MB`,
      generated `#85141`.
    - `--kind chat --tokens 8 --skip-install`: Chat path used the bundled
      tokenizer, auto-ran, auto-exited with code 0, analyzer passed, warm
      average `13.40 sec/token`, peak `620.8 MB`, generated
      `#85141,#236924,#238797,#237669,#227246,#236924,#145728,#237328`.
    - `--kind tokens32 --skip-build --skip-install`: auto-ran and auto-exited
      with code 0, analyzer passed, warm average `12.74 sec/token`, peak
      `587.0 MB`, generated 32 tokens:
      `#85141,#236924,#52119,#12553,#237221,#66447,#237669,#237230,#237214,#3652,#236924,#108,#99844,#70622,#237000,#106241,#35071,#236951,#30886,#236945,#69144,#237512,#201132,#236951,#159734,#56980,#10965,#236951,#125561,#203956,#239542,#139281`.
    - After adding the in-app run-statistics rows, `--kind chat --tokens 8`
      was reinstalled and rerun successfully: analyzer passed, warm average
      `12.50 sec/token`, peak `621.6 MB`, same generated 8-token sequence as
      the prior Chat smoke.
    - 2026-07-04 chunk4 baseline recheck after reinstall:
      `--kind chat --tokens 8` passed, warm average `13.07 sec/token`, peak
      `617.3 MB`, generated
      `#85141,#236924,#238797,#237669,#227246,#236924,#145728,#237328`.
    - 2026-07-04 chunk8 test: replacing decoder chunks with 6 compiled
      8-layer bundles passed asset verification only when the per-bundle limit
      was raised, but the app refused to run with
      `unsafeComputeConfiguration`: 8-layer Seq64 decoder bundles are known to
      fail iPhone execution-plan compilation. The app was restored to chunk4
      and reverified: warm average `13.02 sec/token`, peak `622.9 MB`, same
      generated 8-token sequence.
    - 2026-07-04 speed sweep on restored chunk4 app:
      decoder `CPU+GPU` warm `13.95 sec/token`, peak `590.1 MB`;
      decoder `All` warm `13.85 sec/token`, peak `583.1 MB`;
      decoder `CPU+ANE` warm `51.79 sec/token`, peak `209.6 MB`.
      Keep decoder `All` as the default speed/stability setting; `CPU+ANE` is a
      low-memory fallback only.
    - 2026-07-04 multimodal app-side smoke:
      `--kind multimodal` added and tested. The app created a synthetic
      multimodal placeholder hidden tensor shaped `[1,64,3840]`, fed it through
      the existing 48-layer Seq64 decoder stack, then ran the preferred
      norm+lm_head endpoint. The run auto-exited with analyzer 0 errors, peak
      `584.0 MB`, token total `40.10 sec`, generated `#236812`, and had no LM
      head fallback. Log:
      `.derived-data/CoreMLProbe/DeviceAutomation/20260704-131948/device-console.log`.
    - 2026-07-04 real image embedder smoke:
      RunPod converted
      `gemma4_12b_image_embedder_patches32_int4_block32.mlpackage`, then the Mac
      compiled it into
      `ios/CoreMLProbe/CoreMLProbe/Models/gemma4_12b_image_embedder_patches32_int4_block32.mlmodelc`.
      The compiled endpoint accepts `pixel_values [1,32,6912]` plus
      `image_position_ids [1,32,2]` and emits `image_hidden [1,32,3840]`.
      `--kind image` passed on iPhone with analyzer 0 errors, image embedder
      load `0.2879 sec`, image embedder predict `0.0161 sec`, peak `581.8 MB`,
      token total `34.4571 sec`, generated `#258883`, and no LM head fallback.
      Log:
      `.derived-data/CoreMLProbe/DeviceAutomation/20260704-141011/device-console.log`.
      This uses a 32-patch raw fixture, not the full Hugging Face image
      processor output.
    - 2026-07-04 real audio embedder smoke:
      RunPod converted
      `gemma4_12b_audio_embedder_tokens32_int4_block32.mlpackage`, then the Mac
      compiled it into
      `ios/CoreMLProbe/CoreMLProbe/Models/gemma4_12b_audio_embedder_tokens32_int4_block32.mlmodelc`.
      The compiled endpoint accepts `input_features [1,32,640]` and emits
      `audio_hidden [1,32,3840]`. `--kind audio` passed on iPhone with analyzer
      0 errors, audio embedder load `0.0389 sec`, audio embedder predict
      `0.0027 sec`, peak `581.9 MB`, token total `33.1126 sec`, generated
      `#236770`, and no LM head fallback. Log:
      `.derived-data/CoreMLProbe/DeviceAutomation/20260704-142904/device-console.log`.
    - 2026-07-04 text regression after adding the real `image-smoke` and
      `audio-smoke` paths:
      `--kind probe --tokens 8 --skip-build --skip-install` passed with analyzer
      0 errors, warm average `13.29 sec/token`, peak `587.7 MB`, generated
      `#85141,#236924,#52119,#12553,#237221,#66447,#237669,#237230`.
      Log:
      `.derived-data/CoreMLProbe/DeviceAutomation/20260704-143330/device-console.log`.
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
  Seq64 token window, appends each argmax token, and slides the window. It reuses
  embedding and LM head models across generated tokens, while decoder layers
  still use load/predict/release for memory headroom.
- Keep the fixed Seq64 shape for current device tests.
- Accept prompt IDs via `COREML_PROBE_INPUT_IDS` or `--input-ids=`.
- Accept generated token count via `COREML_PROBE_GENERATE_TOKENS` or
  `--tokens=`. The app clamps the UI to `1...64`, but still defaults to `2`.
- The app now has `Chat` and `Probe` tabs. Chat wraps `generate-token-loop`
  and keeps `Cache = Run end` as the default.
- Endpoint models (embedding and LM head) and decoder layers can use separate
  compute-unit settings:
  - `COREML_PROBE_ENDPOINT_COMPUTE` or `--endpoint-compute=`
  - `COREML_PROBE_DECODER_COMPUTE` or `--decoder-compute=`
  - `COREML_PROBE_COMPUTE` or `--compute=` remains the shared fallback.
- Chat now defaults to endpoint `CPU` and decoder `All`, which is the fastest
  stable measured split so far. Probe still starts conservatively unless launch
  settings override it.
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
- `scripts/runpod_convert_gemma4_coreml_multimodal.py` converts fixed-shape
  Gemma 4 Unified image/audio embedders for smoke tests. The current app uses
  the 32-patch image package and 32-token audio package from this script.
- `scripts/runpod_refresh_gemma4_coreml_endpoint.sh` is the preferred RunPod
  entrypoint for the endpoint quality fix. It converts the corrected norm+lm_head
  endpoint, writes SHA256 sums, and creates a transfer tarball. It now runs
  `scripts/runpod_preflight_gemma4_coreml_endpoint.sh` first by default.
- `scripts/runpod_preflight_gemma4_coreml_endpoint.sh` checks the RunPod model
  path, output/archive writeability, Python dependencies, CUDA visibility, and
  disk space before the long endpoint conversion starts.
- `scripts/setup_runpod_coreml_endpoint_env.sh` installs the RunPod-side Python
  conversion dependencies without replacing torch by default. Use
  `--install-torch --torch-index-url ...` only when the RunPod image lacks a
  CUDA-enabled torch install.
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
  window checks and generated-ID decode. It depends on
  `transformers sentencepiece jinja2` and defaults to raw prompt tokenization.
- `scripts/analyze_coreml_probe_log.py` summarizes pasted/saved Xcode logs and
  can enforce the post-refresh expectations: preferred norm+lm_head endpoint,
  no fallback, expected layer count, generated token count, repeat detection,
  peak-memory ceiling, and per-token timing breakdown.
- `scripts/preflight_coreml_probe_device.sh` is the strict local gate before a
  real-device run. It verifies staged assets, requires the corrected norm+lm_head
  endpoint by default, runs a generic iOS build, and prints the recommended
  device settings plus log analyzer command. Pass `--require-image-embedder`
  and/or `--require-audio-embedder` when preparing multimodal smoke
  regressions.
- `scripts/run_coreml_probe_device_automation.sh` builds or reuses the
  Debug-iphoneos app, finds the connected iPhone with `devicectl`, installs the
  app unless `--skip-install` is passed, launches it with automation environment
  variables, captures `devicectl --console` output, waits for app auto-exit, and
  runs `analyze_coreml_probe_log.py` on the saved log. It accepts `--kind image`
  for the real 32-patch image embedder smoke, `--kind audio` for the real
  32-token audio embedder smoke, and `--kind multimodal-smoke` for the synthetic
  hidden-input decoder/lm_head seam smoke.
- `ios/CoreMLProbe/scripts/run_coreml_probe_device_automation.sh` is a thin
  wrapper for the same root script, so the command also works from the Xcode
  project directory.
- `scripts/import_coreml_probe_endpoint.sh <endpoint-mlpackage-dir-or-tar.gz>`
  is the preferred Mac-side one-shot after copying the RunPod artifact back. It
  runs endpoint refresh without a duplicate build, then runs strict device
  preflight once.
- A simple app icon exists in `Assets.xcassets/AppIcon.appiconset`.
- Chat composer text is tokenized on device with the bundled Gemma tokenizer and
  left-padded to Seq64 when needed. It also accepts pasted `input_ids_last64=...`,
  `input_ids_last4=...`, `input_ids=...`, `#123` IDs, or raw ID lists from the
  message body; pasted IDs take priority over tokenizing the text.
- Chat now defaults to `Max Tokens = 8` and exposes a segmented `Max Tokens`
  preset control for `8 / 16 / 32`. Probe still keeps its free token-count
  stepper for lower-level experiments.
- Chat assistant rows decode generated token IDs through the bundled tokenizer
  when possible, while still showing the generated `#id` chips and raw
  generated-token detail.
- `generate-token-loop` stops early on EOS or turn delimiter token IDs (`#1`,
  `#106`) and still records a `Stop generation` step. It also keeps the
  repeated-token guard for four identical generated tokens.
- Decoder chunk bundles wider than 4 layers are intentionally blocked at app
  launch time. A 2026-07-04 chunk8 experiment confirmed the existing guard is
  still relevant: 8-layer Seq64 bundles should not be used on iPhone.
- Probe has a `Run Retain Sweep` action that compares endpoint `CPU`, decoder
  `All`, `Tokens = 8`, Retain 0 vs Retain 1. Current device evidence shows
  Retain 1 is not worth using: no speed gain and peak memory around `1102 MB`.
- App launch automation accepts `COREML_PROBE_AUTORUN` or `--autorun=`, with
  values like `probe`, `chat`, `stability-sweep`, `retention-sweep`, and
  `speed-sweep`. `COREML_PROBE_AUTO_EXIT=1` or `--auto-exit` exits the app after
  the automated run so `devicectl --console` returns.
- Chat and Probe status now summarize run statistics from recorded steps:
  generated/target tokens, warm average, all-token average, peak memory, stop
  reason, and per-token timing split for decoder load, decoder predict, and LM
  head.

## Recommended next step

The best current cache policy for generation is `Run end`, and the best measured
compute split is endpoint `CPU` with decoder `All`. That combination completed
the latest bundled-tokenizer `Tokens = 8` and `Tokens = 16` runs through all 48
layers without crashing, with peak memory around `605-621 MB`.

1. Keep Chat on endpoint `CPU`, decoder `All`, `Layers = First 48`,
   `Cache = Run end`, Retain 0, `Max Tokens = 8`.
2. Use `scripts/run_coreml_probe_device_automation.sh` for further device runs
   instead of manual Xcode console copy/paste. First install can be slow because
   the app bundle is about `6.8G`; after that use `--skip-build --skip-install`.
3. `Max Tokens = 32` has passed one ceiling probe with peak `587.0 MB`; keep it
   as an explicit option, but keep normal Chat default at 8 because latency is
   still about `12-13 sec/token`.
4. Use `scripts/gemma4_token_helper.py --prompt ...` and `--decode-ids ...` when
   comparing exact token IDs, but normal Chat text now goes through the bundled
   tokenizer.
   - Historical 16-token iPhone test with the old linear-only LM head showed run
     1 sliding from `2,123,4567,106` to
     `253027,253027,253027,253027` by token 6, then run 2 starting from that
     repeated window. At that point top logits were identical and argmax stayed
     `#253027`.
   - A later old-endpoint tokenizer-derived test with `237234,7604,31600,3335`
     confirmed the app used the pasted window, but it still collapsed to
     `#253027` by token 5. The current app has since moved to the corrected
     norm+lm_head endpoint and bundled tokenizer.
5. Do not spend more time on 8-layer decoder chunks for Seq64 on iPhone; they
   are blocked by the app because execution-plan compilation failed in prior
   testing. If probing accelerator behavior, keep endpoint `CPU`; `CPU+ANE` is
   much lower memory but was far slower in the 4-token sweep, so it is a
   fallback path rather than the speed path.
6. Keep `image-smoke` and `audio-smoke` as the current multimodal feasibility
   proof. They run real Gemma 4 image/audio embedders on fixed 32-token
   fixtures and feed those hidden states into the stable 48-layer decoder/lm_head
   path.
7. RunPod is not needed for local app-side loops. If the 2026-07-04 A100 SXM pod
   is still running, it can be stopped/terminated unless the next conversion is
   about to start. Keep the storage for now because it now contains the model,
   venv, conversion logs, and image/audio embedder packages.

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
- The Hugging Face processor uses the `<|image|>` placeholder. In RunPod
  inspection, standard image processing produced 256 image tokens and
  `pixel_values [1,280,6912]` even for very small PIL images, so a full standard
  image prompt does not fit the current Seq64 window.
- Current `image-smoke` deliberately uses a 32-patch raw fixture through the
  real Gemma 4 image embedder, and `audio-smoke` uses a 32-token synthetic audio
  feature fixture through the real Gemma 4 audio embedder. This proves
  runtime/model plumbing, not semantic image/audio understanding quality.
- Next multimodal options are: build real image/audio preprocessing in the app,
  or design a larger/fuller token window if the goal shifts from feasibility
  smoke to more faithful prompting.

## Useful commands

```bash
./scripts/build_coreml_probe_ios.sh
./scripts/build_coreml_probe_ios.sh 'generic/platform=iOS'
./scripts/run_coreml_probe_device_automation.sh --skip-build --kind probe --tokens 1 --min-generated-tokens 1
./scripts/run_coreml_probe_device_automation.sh --skip-build --skip-install --kind chat --tokens 8
./scripts/run_coreml_probe_device_automation.sh --skip-build --kind image --timeout 1800
./scripts/run_coreml_probe_device_automation.sh --skip-build --kind audio --timeout 1800
./scripts/run_coreml_probe_device_automation.sh --skip-build --skip-install --kind tokens32 --timeout 1800
SEQ_LEN=64 ./scripts/preflight_coreml_probe_device.sh --skip-build --require-image-embedder --require-audio-embedder
(cd ios/CoreMLProbe && ./scripts/run_coreml_probe_device_automation.sh --skip-build --skip-install --kind chat --tokens 8)
du -sh ios/CoreMLProbe/CoreMLProbe/Models .derived-data/CoreMLProbe/Build/Products/Debug-iphoneos/CoreMLProbe.app
find ios/CoreMLProbe/CoreMLProbe/Models -maxdepth 1 -name 'gemma4_12b_layers*_decoder_seq64_mask_int4_block32.mlmodelc' -type d | wc -l
```

When committing, stage only the files changed intentionally for the task.
