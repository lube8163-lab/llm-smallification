#!/usr/bin/env python3
"""Convert Gemma 4 text endpoint modules to compressed Core ML packages.

This RunPod-side script complements the per-layer decoder conversion. It emits
the fixed-shape token embedding package and a corrected final endpoint package
that applies the language-model final RMSNorm before lm_head, including Gemma's
final logit softcap.
"""

from __future__ import annotations

import argparse
import gc
import os
import time
from pathlib import Path

import coremltools as ct
import coremltools.optimize.coreml as cto
import numpy as np
import torch
from transformers import AutoModelForImageTextToText


class TextEmbeddingWrapper(torch.nn.Module):
    def __init__(self, language_model: torch.nn.Module):
        super().__init__()
        self.embed_tokens = language_model.embed_tokens

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)


class NormLMHeadWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, language_model: torch.nn.Module):
        super().__init__()
        self.norm = language_model.norm
        self.lm_head = model.lm_head
        text_config = model.config.get_text_config()
        self.final_logit_softcapping = text_config.final_logit_softcapping

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)
        if self.final_logit_softcapping is not None:
            logits = logits / self.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.final_logit_softcapping
        return logits


def package_size(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for filename in files:
            total += os.path.getsize(os.path.join(root, filename))
    return total


def resolve_language_model(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model
    if hasattr(model, "language_model"):
        return model.language_model
    raise AttributeError("could not find model.model.language_model")


def convert_package(
    *,
    name: str,
    wrapper: torch.nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    input_types: list[ct.TensorType],
    output_name: str,
    out_dir: Path,
    quant_config: cto.OptimizationConfig,
    quant_suffix: str,
    quant_kind: str,
    keep_fp16: bool,
    force: bool,
) -> None:
    fp16_path = out_dir / f"{name}_fp16.mlpackage"
    int4_path = out_dir / f"{name}_{quant_suffix}.mlpackage"

    if int4_path.exists() and not force:
        print("skip_existing", int4_path, "bytes", package_size(int4_path), flush=True)
        return

    print("convert_endpoint", name, flush=True)
    wrapper = wrapper.eval().cuda()
    with torch.inference_mode():
        output = wrapper(*example_inputs)
        print("forward_ok", name, tuple(output.shape), output.dtype, flush=True)
        traced = torch.jit.trace(wrapper, example_inputs, strict=False)

    traced = traced.eval().cpu()
    print("trace_ok", name, flush=True)

    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=input_types,
        outputs=[ct.TensorType(name=output_name, dtype=np.float16)],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )

    if keep_fp16:
        mlmodel.save(fp16_path)
        print("saved_fp16", fp16_path, package_size(fp16_path), flush=True)

    if quant_kind == "int4-block":
        qmodel = cto.linear_quantize_weights(mlmodel, config=quant_config)
    else:
        qmodel = cto.palettize_weights(mlmodel, config=quant_config)
    qmodel.save(int4_path)
    print(
        "saved_int4",
        int4_path,
        "bytes",
        package_size(int4_path),
        "gib",
        round(package_size(int4_path) / 1024**3, 3),
        flush=True,
    )

    del qmodel, mlmodel, traced, wrapper, output
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        default="/workspace/gemma12b/models/gemma-4-12b-it-qat-q4_0-unquantized",
    )
    parser.add_argument(
        "--out-dir",
        default="/workspace/gemma12b/coreml-endpoints-seq4-int4",
    )
    parser.add_argument("--seq-len", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument(
        "--quant",
        choices=["int4-block", "palettize4"],
        default="int4-block",
        help=(
            "int4-block: linear int4 per-block (CPU/BNNS on device)."
            " palettize4: 4-bit LUT palettization so the endpoint can run on"
            " the ANE alongside pal4 decoder chunks."
        ),
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=16,
        help="per_grouped_channel group size for --quant palettize4",
    )
    parser.add_argument(
        "--target",
        choices=["all", "embedding", "norm-lm-head"],
        default="all",
    )
    parser.add_argument("--keep-fp16", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("coremltools", ct.__version__, flush=True)
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)
    print("loading", args.model_dir, flush=True)
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_dir,
        dtype=torch.float16,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
    )
    model.eval()
    language_model = resolve_language_model(model)

    print(
        "loaded_sec",
        round(time.time() - t0, 2),
        "cuda_alloc_gb",
        round(torch.cuda.memory_allocated() / 1024**3, 3),
        "softcap",
        model.config.get_text_config().final_logit_softcapping,
        flush=True,
    )

    if args.quant == "int4-block":
        quant_config = cto.OptimizationConfig(
            global_config=cto.OpLinearQuantizerConfig(
                mode="linear_symmetric",
                dtype="int4",
                granularity="per_block",
                block_size=args.block_size,
            )
        )
        quant_suffix = f"int4_block{args.block_size}"
    else:  # palettize4
        quant_config = cto.OptimizationConfig(
            global_config=cto.OpPalettizerConfig(
                mode="kmeans",
                nbits=4,
                granularity="per_grouped_channel",
                group_size=args.group_size,
            )
        )
        quant_suffix = f"pal4_g{args.group_size}"

    if args.target in {"all", "embedding"}:
        example_input_ids = torch.zeros(
            (1, args.seq_len),
            device="cuda",
            dtype=torch.int32,
        )
        convert_package(
            name=f"gemma4_12b_embedding_seq{args.seq_len}",
            wrapper=TextEmbeddingWrapper(language_model),
            example_inputs=(example_input_ids,),
            input_types=[
                ct.TensorType(name="input_ids", shape=example_input_ids.shape, dtype=np.int32)
            ],
            output_name="hidden",
            out_dir=out_dir,
            quant_config=quant_config,
            quant_suffix=quant_suffix,
            quant_kind=args.quant,
            keep_fp16=args.keep_fp16,
            force=args.force,
        )

    if args.target in {"all", "norm-lm-head"}:
        example_hidden = torch.randn(
            (1, 1, model.config.get_text_config().hidden_size),
            device="cuda",
            dtype=torch.float16,
        )
        convert_package(
            name="gemma4_12b_norm_lm_head_1tok",
            wrapper=NormLMHeadWrapper(model, language_model),
            example_inputs=(example_hidden,),
            input_types=[
                ct.TensorType(name="hidden", shape=example_hidden.shape, dtype=np.float16)
            ],
            output_name="logits",
            out_dir=out_dir,
            quant_config=quant_config,
            quant_suffix=quant_suffix,
            quant_kind=args.quant,
            keep_fp16=args.keep_fp16,
            force=args.force,
        )

    print("done", flush=True)


if __name__ == "__main__":
    main()
