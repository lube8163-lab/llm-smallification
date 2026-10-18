# Storage Inventory: 2026-07-04

This records the local Mac storage state after the Gemma 4 12B Core ML iPhone
multimodal smoke work.

## Keep

- `ios/CoreMLProbe/CoreMLProbe/Models`:
  active ignored iOS app assets used for the latest text/image/audio smoke
  runs. Size is about `6.8G`.
- `runpod-artifacts/coreml-multimodal`:
  local copy of the latest image/audio `.mlpackage` files. Small, about `40M`.
- `runpod-artifacts/gemma4-norm-lm-head-endpoint.tar.gz`:
  local backup of the corrected norm+lm_head endpoint package. About `477M`.
- `.derived-data/CoreMLProbe/DeviceAutomation`:
  small real-device logs and analyzer outputs. About `1.3M`.

## Safe Cleanup Candidates

These ignored/generated artifacts were removed from the Mac because they are no
longer the active app assets:

- `.derived-data/CoreMLProbe/Build`: generic/device build products, regeneratable.
- `runpod-artifacts/compiled-seq64-chunk4`: duplicate compiled app assets now
  staged under `ios/CoreMLProbe/CoreMLProbe/Models`.
- `runpod-artifacts/coreml-seq64-chunk4-int4`: local conversion staging; can be
  regenerated from RunPod/model conversion if needed.
- `runpod-artifacts/compiled-seq64-chunk8` and
  `runpod-artifacts/coreml-seq64-chunk8-int4`: failed/abandoned chunk8
  experiment. iPhone execution-plan compilation rejected these bundles.
- `models/google-gemma-4-12b-qat-q4_0`: old local model copy not used by the
  current iOS CoreMLProbe app.
- `external/llama.cpp`: old local external checkout/build tree, not used by the
  current iOS CoreMLProbe app.

## Cleanup Result

Before cleanup:

- Project-adjacent generated data:
  - `.derived-data`: `14G`
  - `runpod-artifacts`: `26G`
  - `models`: `7.1G`
  - `external`: `7.9G`
  - active `ios/CoreMLProbe/CoreMLProbe/Models`: `6.8G`
- Volume free space: `485Gi`

After cleanup:

- Remaining generated data:
  - `.derived-data`: `1.4M` (device logs kept)
  - `runpod-artifacts`: `517M` (latest multimodal packages and endpoint archive kept)
  - active `ios/CoreMLProbe/CoreMLProbe/Models`: `6.8G`
- Removed empty `models` and `external` directories.
- Volume free space: `513Gi`

The active iOS model bundles were not deleted. A strict asset preflight still
passed after cleanup:

```bash
SEQ_LEN=64 ./scripts/preflight_coreml_probe_device.sh --skip-build --require-image-embedder --require-audio-embedder
```

## Latest Device Evidence

- Image smoke log:
  `.derived-data/CoreMLProbe/DeviceAutomation/20260704-141011/device-console.log`
- Audio smoke log:
  `.derived-data/CoreMLProbe/DeviceAutomation/20260704-142904/device-console.log`
- Text regression log after image/audio smoke:
  `.derived-data/CoreMLProbe/DeviceAutomation/20260704-143330/device-console.log`

## Remote State

The RunPod storage mounted at `/workspace/gemma12b` contains the downloaded
23GB model, venv, conversion logs, and the generated image/audio embedder
packages. Keep the network storage if fuller image/audio conversion may continue.
The A100 pod itself can be stopped when not converting.
