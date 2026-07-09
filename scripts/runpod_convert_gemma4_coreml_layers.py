#!/usr/bin/env python3
"""Convert Gemma 4 text decoder layers to compressed Core ML MLProgram packages.

This is a RunPod-side experiment script. It intentionally converts fixed-shape,
cache-free decoder layers first; that gives us an operator/packaging baseline
before attempting an iPhone decode loop with explicit KV cache plumbing.
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


class DecoderLayerMaskWrapper(torch.nn.Module):
    def __init__(self, language_model: torch.nn.Module, layer_idx: int):
        super().__init__()
        self.rotary_emb = language_model.rotary_emb
        self.layer = language_model.layers[layer_idx]
        self.layer_type = language_model.config.layer_types[layer_idx]

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        position_embeddings = self.rotary_emb(x, position_ids, self.layer_type)
        return self.layer(
            x,
            shared_kv_states={},
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
        )


class DecoderLayerChunkMaskWrapper(torch.nn.Module):
    def __init__(self, language_model: torch.nn.Module, layer_indices: list[int]):
        super().__init__()
        self.rotary_emb = language_model.rotary_emb
        self.layer_indices = layer_indices
        self.layers = torch.nn.ModuleList(
            [language_model.layers[layer_idx] for layer_idx in layer_indices]
        )
        self.layer_types = [
            language_model.config.layer_types[layer_idx] for layer_idx in layer_indices
        ]

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer, layer_type in zip(self.layers, self.layer_types):
            position_embeddings = self.rotary_emb(x, position_ids, layer_type)
            x = layer(
                x,
                shared_kv_states={},
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
            )
        return x


def package_size(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for filename in files:
            total += os.path.getsize(os.path.join(root, filename))
    return total


def parse_layers(spec: str, num_layers: int) -> list[int]:
    if spec == "all":
        return list(range(num_layers))

    layers: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            layers.update(range(start, end + 1))
        else:
            layers.add(int(part))

    result = sorted(layers)
    bad = [idx for idx in result if idx < 0 or idx >= num_layers]
    if bad:
        raise ValueError(f"layer index out of range: {bad}")
    return result


def layer_groups(layers: list[int], chunk_size: int) -> list[list[int]]:
    if chunk_size <= 1:
        return [[layer] for layer in layers]

    groups: list[list[int]] = []
    current: list[int] = []
    for layer in layers:
        if current and (layer != current[-1] + 1 or len(current) >= chunk_size):
            groups.append(current)
            current = []
        current.append(layer)

    if current:
        groups.append(current)
    return groups


def decoder_prefix(layer_group: list[int], seq_len: int) -> str:
    if len(layer_group) == 1:
        return f"gemma4_12b_layer{layer_group[0]:02d}_decoder_seq{seq_len}_mask"
    return (
        f"gemma4_12b_layers{layer_group[0]:02d}_{layer_group[-1]:02d}"
        f"_decoder_seq{seq_len}_mask"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        default="/workspace/gemma12b/models/gemma-4-12b-it-qat-q4_0-unquantized",
    )
    parser.add_argument(
        "--out-dir",
        default="/workspace/gemma12b/coreml-layers-seq4-int4",
    )
    parser.add_argument("--layers", default="all")
    parser.add_argument("--seq-len", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument(
        "--quant",
        choices=["int4-block", "palettize4"],
        default="int4-block",
        help=(
            "int4-block: linear int4 per-block (BNNS/CPU-only on device, current"
            " assets). palettize4: 4-bit LUT palettization, the representation"
            " the ANE can execute — used to probe escaping the BNNS"
            " concurrent-plan limit."
        ),
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=16,
        help="per_grouped_channel group size for --quant palettize4",
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
    language_model = model.model.language_model
    language_model.config._attn_implementation = "eager"
    for layer in language_model.layers:
        layer.self_attn.config._attn_implementation = "eager"

    print(
        "loaded_sec",
        round(time.time() - t0, 2),
        "cuda_alloc_gb",
        round(torch.cuda.memory_allocated() / 1024**3, 3),
        flush=True,
    )

    layers = parse_layers(args.layers, len(language_model.layers))
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

    seq_len = args.seq_len
    example_x = torch.randn(1, seq_len, 3840, device="cuda", dtype=torch.float16)
    example_pos = torch.arange(seq_len, device="cuda", dtype=torch.int32).unsqueeze(0)
    example_mask = torch.triu(
        torch.full(
            (1, 1, seq_len, seq_len),
            -65504.0,
            device="cuda",
            dtype=torch.float16,
        ),
        diagonal=1,
    )

    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be >= 1")

    total_int4_size = 0
    converted = 0
    groups = layer_groups(layers, args.chunk_size)
    for group in groups:
        first_layer_idx = group[0]
        last_layer_idx = group[-1]
        attn = language_model.layers[first_layer_idx].self_attn
        prefix = decoder_prefix(group, seq_len)
        fp16_path = out_dir / f"{prefix}_fp16.mlpackage"
        int4_path = out_dir / f"{prefix}_{quant_suffix}.mlpackage"

        if int4_path.exists() and not args.force:
            size = package_size(int4_path)
            total_int4_size += size
            print(
                "skip_existing",
                f"{first_layer_idx}-{last_layer_idx}",
                int4_path,
                "bytes",
                size,
                flush=True,
            )
            continue

        print(
            "convert_group",
            f"{first_layer_idx}-{last_layer_idx}",
            "layers",
            group,
            "chunk_size",
            len(group),
            "type",
            ",".join(language_model.config.layer_types[layer_idx] for layer_idx in group),
            "has_v_proj",
            attn.v_proj is not None,
            "store_full_length_kv",
            attn.store_full_length_kv,
            flush=True,
        )

        if len(group) == 1:
            wrapper = DecoderLayerMaskWrapper(language_model, first_layer_idx).eval().cuda()
        else:
            wrapper = DecoderLayerChunkMaskWrapper(language_model, group).eval().cuda()
        with torch.inference_mode():
            output = wrapper(example_x, example_pos, example_mask)
            print("forward_ok", tuple(output.shape), output.dtype, flush=True)
            traced = torch.jit.trace(
                wrapper,
                (example_x, example_pos, example_mask),
                strict=False,
            )

        traced = traced.eval().cpu()
        print("trace_ok", f"{first_layer_idx}-{last_layer_idx}", flush=True)

        mlmodel = ct.convert(
            traced,
            convert_to="mlprogram",
            inputs=[
                ct.TensorType(name="x", shape=example_x.shape, dtype=np.float16),
                ct.TensorType(
                    name="position_ids",
                    shape=example_pos.shape,
                    dtype=np.int32,
                ),
                ct.TensorType(
                    name="attention_mask",
                    shape=example_mask.shape,
                    dtype=np.float16,
                ),
            ],
            outputs=[ct.TensorType(name="y", dtype=np.float16)],
            minimum_deployment_target=ct.target.iOS18,
            compute_precision=ct.precision.FLOAT16,
        )

        if args.keep_fp16:
            mlmodel.save(fp16_path)
            print("saved_fp16", fp16_path, package_size(fp16_path), flush=True)

        if args.quant == "int4-block":
            qmodel = cto.linear_quantize_weights(mlmodel, config=quant_config)
        else:
            qmodel = cto.palettize_weights(mlmodel, config=quant_config)
        qmodel.save(int4_path)
        int4_size = package_size(int4_path)
        total_int4_size += int4_size
        converted += 1
        print(
            "saved_int4",
            int4_path,
            "bytes",
            int4_size,
            "gib",
            round(int4_size / 1024**3, 3),
            flush=True,
        )

        del qmodel, mlmodel, traced, wrapper, output
        gc.collect()
        torch.cuda.empty_cache()

    print(
        "done",
        "converted",
        converted,
        "groups_seen",
        len(groups),
        "layers_seen",
        len(layers),
        "total_int4_gib",
        round(total_int4_size / 1024**3, 3),
        flush=True,
    )


if __name__ == "__main__":
    main()
