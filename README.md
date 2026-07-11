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
./scripts/compile_coreml_probe_packages.sh
./scripts/prepare_ios_coreml_probe_assets.sh
```

When updating only the corrected generation endpoint, run the RunPod-side
conversion/package wrapper, transfer the resulting endpoint package or archive
back to the Mac, then refresh the app-local endpoint:

```bash
./scripts/setup_runpod_coreml_endpoint_env.sh
./scripts/runpod_refresh_gemma4_coreml_endpoint.sh
./scripts/import_coreml_probe_endpoint.sh <endpoint-mlpackage-dir-or-tar.gz>
```

The RunPod wrapper runs `runpod_preflight_gemma4_coreml_endpoint.sh` first to
check the model path, Python dependencies, CUDA visibility, writable output
paths, and disk space before starting conversion.

Then build the probe app for a simulator:

```bash
./scripts/build_coreml_probe_ios.sh
```

The simulator app can auto-run the probe and print `[CoreMLProbe]` lines:

```bash
xcrun simctl launch --console booted lab.lube8163.CoreMLProbe --autorun
```

For iPhone memory triage, start with CPU-only single-model loads before trying
the full pipeline. Useful scheme arguments or launch environment values:

```bash
--autorun --mode=load-embedding
--autorun --mode=load-decoder
--autorun --mode=load-lm-head
--autorun --mode=load-all-sequential
--autorun --mode=load-decoder-stack --layers=16
--autorun --mode=decoder-stack --layers=16
--autorun --mode=full-sequential
--autorun --mode=full-stack-sequential --layers=16
--autorun --mode=generate-one-token --layers=48 --input-ids=2,123,4567,106
--autorun --mode=generate-token-loop --layers=48 --input-ids=2,123,4567,106 --tokens=2
--autorun --mode=generate-token-loop --layers=48 --input-ids=2,123,4567,106 --tokens=2 --cache-policy=every-8-layers
--autorun --mode=generate-token-loop --layers=48 --input-ids=2,123,4567,106 --tokens=8 --cache-policy=run-end-only
--autorun --mode=generate-token-loop --layers=8 --input-ids=2,123,4567,106 --tokens=1 --cache-policy=run-end-only --decoder-compute=cpuAndGPU --endpoint-compute=cpuOnly
--autorun --mode=generate-token-loop --layers=48 --input-ids=2,123,4567,106 --tokens=8 --cache-policy=run-end-only --endpoint-compute=cpuOnly --decoder-compute=cpuAndGPU
```

```bash
COREML_PROBE_MODE=load-lm-head
COREML_PROBE_COMPUTE=cpuOnly
COREML_PROBE_DECODER_COMPUTE=cpuAndGPU
COREML_PROBE_ENDPOINT_COMPUTE=cpuOnly
COREML_PROBE_LAYERS=16
COREML_PROBE_INPUT_IDS=2,123,4567,106
COREML_PROBE_GENERATE_TOKENS=8
COREML_PROBE_CACHE_POLICY=run-end-only
```

The app defaults to `load-embedding` and `cpuOnly` to avoid loading multiple
large bundles on the first run. Decoder stack modes default to the first 8
layers and can be limited to `1`, `2`, `4`, `8`, `16`, `24`, `32`, `48`, or
`all`. Try `full-sequential` only after the individual loads and synthetic
predictions pass. When additional decoder layer bundles are present in
`Models/`, `load-decoder-stack`, `decoder-stack`, and `full-stack-sequential`
discover them automatically by layer index.

`generate-token-loop` keeps the fixed 4-token window, appends each argmax token,
and slides the window for a very short generation loop. It reuses the embedding
and LM head models across generated tokens, while decoder layers still use the
safer load/predict/release path.

The Chat tab is a probe wrapper over that fixed token window. Plain message text
is not tokenized on device yet. To change the model input from the composer,
paste `input_ids_last4=...`, `input_ids=...`, four `#123`-style IDs, or four raw
IDs; otherwise the current `Token window` field is used and the app shows that
source in the message detail. If generation collapses to the same token in all 4
positions, the runner logs `Repeated input window` before continuing.

For generation quality probes, the preferred endpoint is now
`gemma4_12b_norm_lm_head_1tok_int4_block32.mlmodelc`, produced by
`scripts/runpod_convert_gemma4_coreml_endpoints.py`. It applies the final
language-model RMSNorm and Gemma final logit softcap before lm_head. If that
bundle is not present, the app falls back to the older linear-only
`gemma4_12b_lm_head_1tok_int4_block32.mlmodelc` and logs `LM head fallback`.
With the preferred endpoint present, it logs `LM head target` before loading the
norm+lm_head bundle.

The UI and launch arguments allow separate compute-unit choices for the large
decoder packages and the endpoint packages (embedding and LM head). Use
`COREML_PROBE_DECODER_COMPUTE` / `--decoder-compute=` and
`COREML_PROBE_ENDPOINT_COMPUTE` / `--endpoint-compute=` for this split. The
single `COREML_PROBE_COMPUTE` / `--compute=` setting remains as a shared
fallback. For accelerator probing on iPhone, start with `--layers=1`,
`--layers=8`, then `16`, `32`, and finally `48`, keeping
`--cache-policy=run-end-only` and `--tokens=1` until the smaller run is stable.
The fastest stable measured split is endpoint `CPU` with decoder `CPU+GPU`; on
an iPhone18,3 this completed all 48 layers for 8 generated tokens with about
`316 MB` peak memory and about `13.7 sec/token` after the first token. Endpoint
`All` exceeded the iOS high-water memory limit during endpoint model load, so
endpoint modes that include ANE are hidden in the UI and blocked before load for
endpoint-using modes. The generated token count can be raised up to `32`, but
the default stays at `2` because full 48-layer generation is still slow.

The cache policy controls how often `com.apple.e5rt.e5bundlecache` is removed
after model release. Supported values are `every-model` (default/safest),
`every-4-layers`, `every-8-layers`, `per-token`, and `run-end-only`.

Until an on-device tokenizer is bundled, use the host helper to convert a prompt
into the fixed 4-token app window or decode generated IDs:

```bash
python3 -m pip install transformers sentencepiece jinja2
python3 scripts/gemma4_token_helper.py --prompt "Hello"
python3 scripts/gemma4_token_helper.py --decode-ids 253027,253027
```

Copy the helper's `input_ids_last4=...` line into the Chat composer or the
`Token window` field to make the prompt affect generation. The helper defaults
to the raw prompt tokens because a full chat template often leaves only the
assistant-prefix tokens in the 4-token window; pass `--chat-template` only when
you explicitly want to inspect that template form.

After a device run, save the Xcode console output and check the endpoint and
generation summary:

```bash
./scripts/analyze_coreml_probe_log.py coremlprobe.log \
  --require-norm-lm-head \
  --fail-on-repeat \
  --expect-layers 48 \
  --min-generated-tokens 8 \
  --max-peak-mb 350
```

The lower-level refresh and preflight scripts remain available when you want to
run the steps separately:

```bash
./scripts/refresh_coreml_probe_endpoint.sh <endpoint-mlpackage-dir-or-tar.gz>
./scripts/preflight_coreml_probe_device.sh
```

For an actual iPhone, open `ios/CoreMLProbe/CoreMLProbe.xcodeproj` in Xcode,
select a signing team, choose the device, and run the `CoreMLProbe` scheme.
The copied `.mlmodelc` bundles remain ignored by git.
When measuring near the memory limit, disable View Debugging and other optional
scheme diagnostics so Xcode does not inject extra debugging libraries.

## Local control API

The iPhone chat API is disabled by default. Enable it only for a trusted local
session from the app's Debug tab. Each activation creates a new bearer token,
all endpoints require that token, and the server stops when the app enters the
background. Diagnostic log endpoints are unavailable in Release builds.

See [`docs/local-api.md`](docs/local-api.md) for the authenticated curl flow and
[`SECURITY.md`](SECURITY.md) for the public-repository security model.

## Model files

Model weights and compiled Core ML bundles are intentionally excluded from this
Git repository. The validated practical-chat Core ML packages are published at
[`lube8163/gemma-4-12b-coreml-iphone-practical-chat`](https://huggingface.co/lube8163/gemma-4-12b-coreml-iphone-practical-chat).

Download, checksum-verify, compile, and stage them with:

```bash
./scripts/download_hf_gemma4_coreml_models.sh
```

See [`docs/model-distribution.md`](docs/model-distribution.md) for the artifact
layout, licensing notices, and publication policy.

## License

Repository scripts and documentation are MIT licensed. Upstream models,
tokenizers, and third-party projects remain governed by their own licenses and
terms.
