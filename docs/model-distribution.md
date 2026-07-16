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

1. Create the Hugging Face model repository as private.
2. Upload one minimal, checksum-verified asset group through Xet.
3. Test a clean download, compile, import, and device run.
4. Complete the model card and license/notice files.
5. Decide whether access should be public or gated.
6. Only then add the model repository URL to the GitHub README and article.
