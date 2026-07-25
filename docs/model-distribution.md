# Model distribution

The practical-chat package set is published separately at
[`lube8163/gemma-4-12b-coreml-iphone-practical-chat`](https://huggingface.co/lube8163/gemma-4-12b-coreml-iphone-practical-chat).

The LLaDA-MoE dLLM/GGUF side experiment is now maintained separately at
[`lube8163-lab/llada-iphone-dllm`](https://github.com/lube8163-lab/llada-iphone-dllm).

## Recommendation

Keep source code, conversion scripts, and documentation in GitHub. Publish
converted model artifacts in a separate Hugging Face model repository rather
than adding them to this Git history.

The current local CoreMLProbe model directory is approximately 18 GB. Hugging
Face model repositories are designed for large binary artifacts and use Xet
storage, while a normal source repository would make every clone and history
operation unnecessarily heavy.

## Artifact format

Prefer distributable `.mlpackage` sources (or deterministic archives of those
packages) over local `.mlmodelc` output. `.mlmodelc` is a compiled deployment
artifact tied more closely to the Core ML compiler and OS toolchain. Users can
compile `.mlpackage` files locally with `coremlcompiler` or import them through
the repository scripts.

For each artifact include:

- exact source model and revision;
- conversion script commit;
- quantization, sequence length, layer range, and compute assumptions;
- SHA-256 checksum and uncompressed size;
- minimum iOS/Core ML version;
- a manifest mapping each filename to its role;
- a short validation result from the matching iPhone configuration.

Large model families should be separated into logical groups such as endpoint,
prefill, decode, multimodal embedders, and optional MTP drafter assets. This
lets users download only the path they intend to run.

## Published layout

```text
models/
  endpoints/               embedding and norm/lm_head
  kv/
    prefill/               48 single-layer Seq320 packages
    decode/                48 single-layer seq=1 packages
  multimodal/              image and audio embedders
  speculative/
    drafter/               mixed pal4-body/int8-head MTP step
    verify/                48 single-layer seq=4 packages
  fused/
    kv/                    selected layers00...05 prefill/verify pair
```

The default `practical` download profile excludes `speculative/` and `fused/`
to keep the original chat setup smaller. Set `HF_MODEL_PROFILE=speculative`
to download, verify, compile, and stage every published package.

Only the measured-useful first six-layer fusion is distributed. The second
group (layers 06...11) made the same 24-token iPhone 17 run 5.4% slower and
increased peak memory, so it remains a local conversion experiment. Compiled
`.mlmodelc` bundles, device logs, Core ML execution-plan caches, and the
superseded reversed-input MTP drafter are never published.

## License and notices

Gemma 4 is published under Apache License 2.0. A repository containing converted
Gemma 4 weights should therefore include:

- a copy of the Apache 2.0 license;
- retained copyright and attribution notices from the source model;
- prominent notices identifying the files as modified/converted;
- a model card that links to the upstream Google model and its license;
- separate notices for any tokenizer or third-party component with different
  terms.

Do not label the converted weights as solely covered by this code repository's
MIT license. The code and documentation can remain MIT while the model
repository carries the upstream model license and conversion notices.

## Publication sequence

1. Validate package count, names, model interfaces, and selected source paths.
2. Rebuild the repository-wide `SHA256SUMS` without removing existing entries.
3. Upload portable `.mlpackage` sources and model-card metadata through Xet.
4. Test a clean profile download, checksum verification, compilation, and
   app staging.
5. Record the matching conversion-code commit and device result.
6. Do not publish slower, incorrect, compiled, or device-specific artifacts.
