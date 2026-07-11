# Notices

This repository contains modified and converted model artifacts derived from:

- Gemma 4 12B instruction-tuned QAT model by Google:
  `google/gemma-4-12B-it-qat-q4_0-unquantized`
- Upstream weight-file revision:
  `58540658b6c08edab2ddc1fbde7f28cc9987ced3`
- Upstream model page and license information:
  <https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-unquantized>
  and <https://ai.google.dev/gemma/apache_2>

The upstream model is licensed under the Apache License, Version 2.0. The
artifacts in this repository have been modified through graph decomposition,
fixed-shape tracing, precision changes, 4-bit palettization or blockwise
quantization, and conversion to Core ML packages.

Conversion scripts and the accompanying iPhone application are available at:
<https://github.com/lube8163-lab/llm-smallification>.

No endorsement by Google or Apple is implied.
