# LLM Smallification

Experiments for pushing a 12B-class Gemma 4 model toward an 8 GB iPhone target
with Core ML/Core AI style splitting, sequential loading, and aggressive weight
compression.

The repository tracks scripts, notes, patches, and experiment reports. It does
not include model weights, GGUF files, Core ML packages, compiled models, or
RunPod artifacts.

## Current Focus

- Baseline `llama.cpp`/GGUF behavior on an 8 GB Apple Silicon Mac.
- Convert Gemma 4 12B text components to Core ML MLProgram packages.
- Compress decoder layers, embedding, and lm head with CoreMLTools int4
  blockwise quantization.
- Prepare an iOS harness that can measure load, prediction, release, and memory
  behavior on an actual 8 GB iPhone.

## Key Results

- Full Gemma 4 12B QAT-unquantized BF16 reference load succeeded on RunPod A100.
- All 48 text decoder layers converted to fixed-shape Core ML MLProgram
  packages and compressed with int4 block32.
- Decoder layer package total: about 5.715 GiB.
- Text-only package estimate with embedding and lm head: about 6.77 GiB.
- Selected int4 packages compiled and ran through local macOS Core ML runtime.

See [docs/2026-06-27-runpod-coreml-probe.md](docs/2026-06-27-runpod-coreml-probe.md)
for the detailed run log and measurements.

## Layout

- `docs/`: experiment plans and measured results.
- `scripts/`: local and RunPod helper scripts.
- `patches/`: patch files for external probes such as `llama.cpp` SwiftUI.
- `ios/CoreMLProbe/`: an iOS SwiftUI harness for measuring Core ML load,
  prediction, release, and memory behavior.

Ignored local/output directories include `models/`, `logs/`, `external/`, and
`runpod-artifacts/`.

## iOS Core ML Probe

After copying selected compiled Core ML bundles from RunPod into
`runpod-artifacts/compiled`, stage the app-local model assets:

```bash
./scripts/prepare_ios_coreml_probe_assets.sh
```

Then build the probe app for a simulator:

```bash
./scripts/build_coreml_probe_ios.sh
```

The simulator app can auto-run the probe and print `[CoreMLProbe]` lines:

```bash
xcrun simctl launch --console booted lab.lube8163.CoreMLProbe --autorun
```

For an actual iPhone, open `ios/CoreMLProbe/CoreMLProbe.xcodeproj` in Xcode,
select a signing team, choose the device, and run the `CoreMLProbe` scheme.
The copied `.mlmodelc` bundles remain ignored by git.

## License

Repository scripts and documentation are MIT licensed. Upstream models,
tokenizers, and third-party projects remain governed by their own licenses and
terms.
