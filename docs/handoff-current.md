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
- In `generate-one-token`, most time is model load/release overhead:
  about `63.7 sec` of timed steps were package loads, while all 48 decoder
  predictions totaled about `1.4 sec`.
- Earlier 24-layer crash was traced to Core ML/E5RT runtime cache pressure. The
  app now clears `com.apple.e5rt.e5bundlecache` at run start and after each model
  release.

## Current state

- `generate-one-token` is implemented and has passed on iPhone CPU-only with
  `Layers = First 48`.
- Keep the fixed 4-token shape.
- Accept prompt IDs via `COREML_PROBE_INPUT_IDS` or `--input-ids=`.
- Run embedding -> selected decoder layers -> last-token LM head -> argmax.
- Log `Prompt IDs` and `Next token`.
- A simple app icon exists in `Assets.xcassets/AppIcon.appiconset`.

## Recommended next step

Move toward a minimally usable text loop:

1. Add tokenizer/prompt formatting or a host-side helper that feeds known token
   IDs.
2. Implement a repeated one-token loop for a very short fixed window.
3. Decide whether to keep model packages hot, group layer residency, or keep the
   strict load/predict/release path for memory headroom.
4. Re-test CPU-only first before accelerator experiments.

## Useful commands

```bash
./scripts/build_coreml_probe_ios.sh
./scripts/build_coreml_probe_ios.sh 'generic/platform=iOS'
du -sh ios/CoreMLProbe/CoreMLProbe/Models .derived-data/CoreMLProbe/Build/Products/Debug-iphoneos/CoreMLProbe.app
find ios/CoreMLProbe/CoreMLProbe/Models -maxdepth 1 -name 'gemma4_12b_layer*_decoder_seq4_mask_int4_block32.mlmodelc' -type d | wc -l
```

When committing, stage only the files changed intentionally for the task.
