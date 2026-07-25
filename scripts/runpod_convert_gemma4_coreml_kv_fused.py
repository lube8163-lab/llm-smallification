#!/usr/bin/env python3
"""Convert six Gemma 4 decoder layers into one KV-cache Core ML package.

The single-layer exporter creates 48 prefill and 48 speculative-verify
packages. Device-specific Core ML execution plans for those 96 packages can
consume tens of gigabytes. Gemma 4 repeats five sliding-attention layers plus
one full-attention layer, so this exporter keeps that natural six-layer period
inside one ML Program:

  prefill: x, position_ids, attention_mask
      -> y, (k_block_N, v_block_N) for six layers
  verify: x, position_ids, attention_mask,
          (k_cache_N, v_cache_N) for six layers
      -> y, (k_new_N, v_new_N) for six layers

Across the full 48-layer stack this reduces 96 execution plans to 16 without
changing weights, quantization, KV geometry, or target logits.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import coremltools as ct
import coremltools.optimize.coreml as cto
import numpy as np
import torch
from transformers import AutoModelForImageTextToText

from runpod_convert_gemma4_coreml_kv import (
    ConcatCache,
    RecordCache,
    cached_causal_mask,
    causal_mask,
    convert_one,
    layer_kv_shape,
)


FUSED_LAYER_COUNT = 6


class PrefillSixLayerWrapper(torch.nn.Module):
    def __init__(self, language_model, layer_start: int):
        super().__init__()
        self.rotary_emb = language_model.rotary_emb
        self.layers = torch.nn.ModuleList(
            language_model.layers[layer_start : layer_start + FUSED_LAYER_COUNT]
        )
        self.layer_types = language_model.config.layer_types[
            layer_start : layer_start + FUSED_LAYER_COUNT
        ]

    def forward(self, x, position_ids, attention_mask):
        blocks = []
        for layer, layer_type in zip(self.layers, self.layer_types):
            cache = RecordCache()
            position_embeddings = self.rotary_emb(x, position_ids, layer_type)
            x = layer(
                x,
                shared_kv_states={},
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=cache,
            )
            blocks.extend([cache.k, cache.v])
        return (x, *blocks)


class DecodeSixLayerWrapper(torch.nn.Module):
    def __init__(self, language_model, layer_start: int):
        super().__init__()
        self.rotary_emb = language_model.rotary_emb
        self.layers = torch.nn.ModuleList(
            language_model.layers[layer_start : layer_start + FUSED_LAYER_COUNT]
        )
        self.layer_types = language_model.config.layer_types[
            layer_start : layer_start + FUSED_LAYER_COUNT
        ]

    def forward(
        self,
        x,
        position_ids,
        attention_mask,
        k_cache_0,
        v_cache_0,
        k_cache_1,
        v_cache_1,
        k_cache_2,
        v_cache_2,
        k_cache_3,
        v_cache_3,
        k_cache_4,
        v_cache_4,
        k_cache_5,
        v_cache_5,
    ):
        cache_inputs = (
            (k_cache_0, v_cache_0),
            (k_cache_1, v_cache_1),
            (k_cache_2, v_cache_2),
            (k_cache_3, v_cache_3),
            (k_cache_4, v_cache_4),
            (k_cache_5, v_cache_5),
        )
        blocks = []
        for layer, layer_type, (k_cache, v_cache) in zip(
            self.layers, self.layer_types, cache_inputs
        ):
            cache = ConcatCache(k_cache, v_cache)
            position_embeddings = self.rotary_emb(x, position_ids, layer_type)
            x = layer(
                x,
                shared_kv_states={},
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=cache,
            )
            blocks.extend([cache.k, cache.v])
        return (x, *blocks)


class FullSixLayerWrapper(torch.nn.Module):
    """Cache-free reference for the fused wrapper validation."""

    def __init__(self, language_model, layer_start: int):
        super().__init__()
        self.rotary_emb = language_model.rotary_emb
        self.layers = torch.nn.ModuleList(
            language_model.layers[layer_start : layer_start + FUSED_LAYER_COUNT]
        )
        self.layer_types = language_model.config.layer_types[
            layer_start : layer_start + FUSED_LAYER_COUNT
        ]

    def forward(self, x, position_ids, attention_mask):
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


@torch.inference_mode()
def validate_group(language_model, layer_start, dtype, device, decode_seq):
    hidden = language_model.config.hidden_size
    prefill_len = 4
    seq = prefill_len + decode_seq
    x = torch.randn(1, seq, hidden, device=device, dtype=dtype) * 0.5
    pos = torch.arange(seq, device=device, dtype=torch.int32).unsqueeze(0)

    reference = FullSixLayerWrapper(language_model, layer_start).eval()
    y_ref = reference(x, pos, causal_mask(seq, dtype, device))

    prefill = PrefillSixLayerWrapper(language_model, layer_start).eval()
    prefill_outputs = prefill(
        x[:, :prefill_len],
        pos[:, :prefill_len],
        causal_mask(prefill_len, dtype, device),
    )
    y_pre = prefill_outputs[0]

    s_max = seq
    cache_inputs = []
    for offset in range(FUSED_LAYER_COUNT):
        layer_idx = layer_start + offset
        kv_heads, head_dim = layer_kv_shape(language_model, layer_idx)
        k_cache = torch.zeros(
            1, kv_heads, s_max, head_dim, device=device, dtype=dtype
        )
        v_cache = torch.zeros_like(k_cache)
        k_cache[:, :, :prefill_len] = prefill_outputs[1 + 2 * offset]
        v_cache[:, :, :prefill_len] = prefill_outputs[2 + 2 * offset]
        cache_inputs.extend([k_cache, v_cache])

    decode = DecodeSixLayerWrapper(language_model, layer_start).eval()
    y_decode = decode(
        x[:, prefill_len:],
        pos[:, prefill_len:],
        cached_causal_mask(
            query_len=decode_seq,
            cache_len=s_max,
            written=prefill_len,
            dtype=dtype,
            device=device,
        ),
        *cache_inputs,
    )[0]

    max_err = max(
        (y_pre - y_ref[:, :prefill_len]).abs().max().item(),
        (y_decode - y_ref[:, prefill_len:]).abs().max().item(),
    )
    scale = y_ref.abs().max().item()
    rel = max_err / max(scale, 1e-6)
    layer_end = layer_start + FUSED_LAYER_COUNT - 1
    print(
        f"validate layers{layer_start:02d}_{layer_end:02d} "
        f"decode_seq={decode_seq} maxAbsErr={max_err:.5f} "
        f"refMax={scale:.2f} rel={rel:.5f}",
        flush=True,
    )
    if rel > 5e-3:
        raise SystemExit(
            f"validation failed for layers {layer_start}...{layer_end}: rel {rel}"
        )


def selected_group_starts(raw_groups: str, layer_count: int) -> list[int]:
    all_starts = list(range(0, layer_count, FUSED_LAYER_COUNT))
    if raw_groups == "all":
        return all_starts
    selected = []
    for part in raw_groups.split(","):
        group = int(part)
        if group in all_starts:
            selected.append(group)
        elif 0 <= group < len(all_starts):
            selected.append(all_starts[group])
        else:
            raise ValueError(
                f"group {group} is neither a layer start {all_starts} "
                f"nor a group index 0...{len(all_starts) - 1}"
            )
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        default="/workspace/gemma12b/models/gemma-4-12b-it-qat-q4_0-unquantized",
    )
    parser.add_argument(
        "--out-dir",
        default="/workspace/gemma12b/coreml-layers-kv-fused6-pal4",
    )
    parser.add_argument(
        "--groups",
        default="all",
        help="comma-separated group indices/layer starts, or all",
    )
    parser.add_argument("--prefill-seq", type=int, default=320)
    parser.add_argument("--s-max", type=int, default=512)
    parser.add_argument("--decode-seq", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument(
        "--target", choices=["all", "prefill", "decode"], default="all"
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--skip-validate", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.decode_seq < 1:
        parser.error("--decode-seq must be at least 1")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        "coremltools",
        ct.__version__,
        "torch",
        torch.__version__,
        "cuda",
        torch.cuda.is_available(),
        flush=True,
    )
    started = time.time()
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_dir,
        dtype=torch.float16,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
    )
    model.eval()
    language_model = model.model.language_model
    config = language_model.config
    assert getattr(config, "num_kv_shared_layers", 0) == 0
    assert config.num_hidden_layers % FUSED_LAYER_COUNT == 0
    assert args.s_max + args.decode_seq <= config.sliding_window
    print("loaded_sec", round(time.time() - started, 2), flush=True)

    group_starts = selected_group_starts(args.groups, config.num_hidden_layers)
    device, dtype = "cuda", torch.float16
    if not args.skip_validate:
        for layer_start in group_starts:
            validate_group(
                language_model, layer_start, dtype, device, args.decode_seq
            )
    if args.validate_only:
        print("validation ok", flush=True)
        return

    quant_config = cto.OptimizationConfig(
        global_config=cto.OpPalettizerConfig(
            nbits=4,
            mode="kmeans",
            granularity="per_grouped_channel",
            group_size=args.group_size,
        )
    )

    hidden = config.hidden_size
    s_pre, s_max, query = args.prefill_seq, args.s_max, args.decode_seq
    for layer_start in group_starts:
        layer_end = layer_start + FUSED_LAYER_COUNT - 1
        output_suffixes = [
            item
            for layer_idx in range(layer_start, layer_end + 1)
            for item in (f"k_{{kind}}_{layer_idx:02d}", f"v_{{kind}}_{layer_idx:02d}")
        ]

        if args.target in {"all", "prefill"}:
            x = torch.randn(1, s_pre, hidden, device=device, dtype=dtype) * 0.1
            pos = torch.arange(
                s_pre, device=device, dtype=torch.int32
            ).unsqueeze(0)
            mask = causal_mask(s_pre, dtype, device)
            convert_one(
                name=(
                    f"gemma4_12b_layers{layer_start:02d}_{layer_end:02d}_"
                    f"prefill_seq{s_pre}_kv_pal4_g{args.group_size}"
                ),
                wrapper=PrefillSixLayerWrapper(language_model, layer_start),
                example_inputs=(x, pos, mask),
                input_types=[
                    ct.TensorType(name="x", shape=x.shape, dtype=np.float16),
                    ct.TensorType(
                        name="position_ids", shape=pos.shape, dtype=np.int32
                    ),
                    ct.TensorType(
                        name="attention_mask", shape=mask.shape, dtype=np.float16
                    ),
                ],
                output_names=["y"]
                + [name.format(kind="block") for name in output_suffixes],
                out_dir=out_dir,
                quant_config=quant_config,
                force=args.force,
            )

        if args.target in {"all", "decode"}:
            x = torch.randn(1, query, hidden, device=device, dtype=dtype) * 0.1
            pos = torch.arange(
                s_pre, s_pre + query, device=device, dtype=torch.int32
            ).unsqueeze(0)
            mask = cached_causal_mask(
                query_len=query,
                cache_len=s_max,
                written=s_pre,
                dtype=dtype,
                device=device,
            )
            cache_tensors = []
            cache_types = []
            for offset, layer_idx in enumerate(
                range(layer_start, layer_end + 1)
            ):
                kv_heads, head_dim = layer_kv_shape(language_model, layer_idx)
                k_cache = torch.randn(
                    1,
                    kv_heads,
                    s_max,
                    head_dim,
                    device=device,
                    dtype=dtype,
                ) * 0.1
                v_cache = torch.randn_like(k_cache) * 0.1
                cache_tensors.extend([k_cache, v_cache])
                cache_types.extend(
                    [
                        ct.TensorType(
                            name=f"k_cache_{layer_idx:02d}",
                            shape=k_cache.shape,
                            dtype=np.float16,
                        ),
                        ct.TensorType(
                            name=f"v_cache_{layer_idx:02d}",
                            shape=v_cache.shape,
                            dtype=np.float16,
                        ),
                    ]
                )
            if query == 1:
                operation = f"decode_kv{s_max}"
            else:
                operation = f"verify_seq{query}_kv{s_max}"
            convert_one(
                name=(
                    f"gemma4_12b_layers{layer_start:02d}_{layer_end:02d}_"
                    f"{operation}_pal4_g{args.group_size}"
                ),
                wrapper=DecodeSixLayerWrapper(language_model, layer_start),
                example_inputs=(x, pos, mask, *cache_tensors),
                input_types=[
                    ct.TensorType(name="x", shape=x.shape, dtype=np.float16),
                    ct.TensorType(
                        name="position_ids", shape=pos.shape, dtype=np.int32
                    ),
                    ct.TensorType(
                        name="attention_mask", shape=mask.shape, dtype=np.float16
                    ),
                    *cache_types,
                ],
                output_names=["y"]
                + [name.format(kind="new") for name in output_suffixes],
                out_dir=out_dir,
                quant_config=quant_config,
                force=args.force,
            )

    print("done", flush=True)


if __name__ == "__main__":
    main()
