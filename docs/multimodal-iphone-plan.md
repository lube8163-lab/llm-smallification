# Multimodal iPhone Plan

Date: 2026-07-04

## Goal

Prove that a Gemma 4 12B Unified-style multimodal input path can run on a
low-memory iPhone at least as a smoke test. Speed is no longer the primary goal
for this branch of work.

Success means:

- an image or audio-derived Core ML endpoint produces decoder-width hidden
  states;
- those hidden states can be fed into the existing Seq64 decoder stack;
- the existing norm+lm_head endpoint produces at least one generated token;
- the run finishes on device without memory termination.

## Current Baseline

- Text path is working with endpoint `CPU`, decoder `All`, 48 layers, Seq64,
  4-layer decoder chunks, cache `Run end`, retain 0.
- Current text-only chat speed is about `13 sec/token`; this is acceptable for
  the multimodal smoke goal.
- Decoder chunks wider than 4 layers are not a current path forward. The
  8-layer Seq64 chunk experiment was blocked by the app because iPhone
  execution-plan compilation failed in prior testing.

## 2026-07-04 Image And Audio Embedder Smoke Results

The app now has three multimodal probe modes:

- `multimodal-smoke`: synthetic `[1,64,3840]` hidden tensor. This remains a
  decoder-input seam regression test.
- `image-smoke`: real converted Gemma 4 Unified image embedder endpoint for a
  fixed 32-patch raw image fixture.
- `audio-smoke`: real converted Gemma 4 Unified audio embedder endpoint for a
  fixed 32-token audio feature fixture.

The important current proofs are:

`raw patch fixture -> Gemma4 embed_vision Core ML -> image_hidden -> 48 decoder layers -> norm+lm_head`

`audio feature fixture -> Gemma4 embed_audio Core ML -> audio_hidden -> 48 decoder layers -> norm+lm_head`

Converted image package:

- RunPod model: `google/gemma-4-12B-it-qat-q4_0-unquantized`
- Converter: `scripts/runpod_convert_gemma4_coreml_multimodal.py`
- Target package:
  `gemma4_12b_image_embedder_patches32_int4_block32.mlpackage`
- Local compiled bundle:
  `ios/CoreMLProbe/CoreMLProbe/Models/gemma4_12b_image_embedder_patches32_int4_block32.mlmodelc`
- Size: about `39M`
- Core ML contract:
  - `pixel_values`: `tensor<fp32, [1,32,6912]>`
  - `image_position_ids`: `tensor<int32, [1,32,2]>`
  - `image_hidden`: `tensor<fp16, [1,32,3840]>`

Real-device result:

- Command:
  `SEQ_LEN=64 ./scripts/run_coreml_probe_device_automation.sh --skip-build --kind image --timeout 1800`
- Log:
  `.derived-data/CoreMLProbe/DeviceAutomation/20260704-141011/device-console.log`
- Analyzer: `0 error(s), 0 warning(s)`
- Compute: endpoint `CPU`, decoder `All`, layers `first-48`, cache
  `run-end-only`, Seq64
- Image embedder load: `0.2879 sec`, memory `79.8 MB`
- Image embedder predict: `0.0161 sec`, memory `80.1 MB`
- Inserted hidden positions: `32...63`
- Generated token: `#258883`, logit `23.359`
- Token total: `34.4571 sec`
- Peak memory: `581.8 MB`
- Preferred `gemma4_12b_norm_lm_head_1tok_int4_block32` endpoint was selected;
  no LM head fallback occurred.

Converted audio package:

- Converter: `scripts/runpod_convert_gemma4_coreml_multimodal.py`
- Target package:
  `gemma4_12b_audio_embedder_tokens32_int4_block32.mlpackage`
- Local compiled bundle:
  `ios/CoreMLProbe/CoreMLProbe/Models/gemma4_12b_audio_embedder_tokens32_int4_block32.mlmodelc`
- Size: about `1.3M`
- Core ML contract:
  - `input_features`: `tensor<fp32, [1,32,640]>`
  - `audio_hidden`: `tensor<fp16, [1,32,3840]>`

Real-device audio result:

- Command:
  `SEQ_LEN=64 ./scripts/run_coreml_probe_device_automation.sh --kind audio --timeout 1800`
- Log:
  `.derived-data/CoreMLProbe/DeviceAutomation/20260704-142904/device-console.log`
- Analyzer: `0 error(s), 0 warning(s)`
- Compute: endpoint `CPU`, decoder `All`, layers `first-48`, cache
  `run-end-only`, Seq64
- Audio embedder load: `0.0389 sec`, memory `33.4 MB`
- Audio embedder predict: `0.0027 sec`, memory `33.7 MB`
- Inserted hidden positions: `32...63`
- Generated token: `#236770`, logit `26.094`
- Token total: `33.1126 sec`
- Peak memory: `581.9 MB`
- Preferred norm+lm_head endpoint was selected; no LM head fallback occurred.

Regression check after adding the real image/audio smoke paths:

- Command:
  `SEQ_LEN=64 ./scripts/run_coreml_probe_device_automation.sh --skip-build --skip-install --kind probe --tokens 8 --timeout 1800`
- Log:
  `.derived-data/CoreMLProbe/DeviceAutomation/20260704-143330/device-console.log`
- Analyzer: `0 error(s), 0 warning(s)`
- Generated:
  `#85141,#236924,#52119,#12553,#237221,#66447,#237669,#237230`
- Warm average: `13.29 sec/token`
- Peak memory: `587.7 MB`

This is enough to say the iPhone app can run real Gemma 4 Unified image and
audio embedder Core ML endpoints, feed their 3840-wide hidden states into the
existing 48-layer decoder stack, and finish the preferred norm+lm_head path
inside the current memory envelope.

## Important Limitation

This is not yet a full standard image or audio prompt. The Hugging Face
processor for `<|image|>` produced 256 image tokens and `pixel_values` shaped
`[1,280,6912]` in the RunPod inspection, even when the input PIL image was very
small. That does not fit the current Seq64 decoder window.

The current proof is deliberately a micro-image smoke:

- 32 raw patches are passed through the real Gemma 4 image embedder.
- 32 synthetic audio feature vectors are passed through the real Gemma 4 audio
  embedder.
- The resulting 32 modality hidden vectors are inserted into a Seq64 hidden
  tensor.
- The decoder/lm_head path is real and uses the same stable 48-layer backend.

Next work should decide whether to keep this as the official low-memory smoke,
add a smaller custom processor path, or export a larger Seq/token configuration
for fuller image coverage.

## RunPod State

The 2026-07-04 conversion run used the user-deployed A100 SXM pod with the
`gemma12b~~` storage:

- Pod: A100 SXM 80GB instance deployed from the RunPod console
- GPU: `NVIDIA A100-SXM4-80GB`
- Console cost observed: `$1.49/hr`
- SSH details were intentionally omitted from this public-facing note; use the
  current RunPod console if the pod is redeployed.
- Working storage root: `/workspace/gemma12b`
- Venv: `/workspace/gemma12b/venv`
- Model cache/path:
  `/workspace/gemma12b/models/gemma-4-12b-it-qat-q4_0-unquantized`
- Converted image package:
  `/workspace/gemma12b/coreml-multimodal-int4/gemma4_12b_image_embedder_patches32_int4_block32.mlpackage`
- Converted audio package:
  `/workspace/gemma12b/coreml-multimodal-int4/gemma4_12b_audio_embedder_tokens32_int4_block32.mlpackage`
- Conversion log:
  `/workspace/gemma12b/reports/convert-image-embedder-patches32-fp32.log`
- Audio conversion log:
  `/workspace/gemma12b/reports/convert-audio-embedder-tokens32-fp32.log`

Recommendation:

- Stop or terminate the A100 pod while doing local app/design/docs work.
- Keep the storage for now if another fuller-image/audio-processor conversion will
  happen soon; it now contains the 23GB model download, venv, logs, and the
  generated image/audio embedder packages.
- Do not delete older 50GB volumes until they are audited or attached and
  inspected.

## Implementation Direction

The existing app expects a hidden tensor shaped `[1, seq, 3840]` before the
decoder stack. Text uses:

`input_ids -> text embedding mlmodelc -> hidden -> 48 decoder layers -> norm+lm_head`

The current image/audio smoke paths target the same seam:

`pixel_values + image_position_ids -> image embedder mlmodelc -> image_hidden -> 48 decoder layers -> norm+lm_head`

`input_features -> audio embedder mlmodelc -> audio_hidden -> 48 decoder layers -> norm+lm_head`

Practical next order:

1. Keep `image-smoke` and `audio-smoke` as the primary multimodal feasibility
   proof.
2. Keep `multimodal-smoke` only as a synthetic decoder-input regression mode.
3. Use asset verification with `--require-image-embedder` and
   `--require-audio-embedder` before multimodal regressions.
4. If continuing multimodal coverage, experiment with fuller processor outputs
   only if Seq/token budget is increased.
5. Preserve the stable text path while adding modality endpoints.

## Notes

- This is a feasibility branch, not a UX or speed branch.
- Avoid re-exporting decoder chunks unless a separate reason appears; the
  current chunk4 decoder stack is the known-stable backend.
- The image-smoke result proves runtime/model plumbing, not semantic image
  understanding quality.
