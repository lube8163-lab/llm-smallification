#!/usr/bin/env python3
"""Convert Gemma 4 decoder layers to KV-cache Core ML packages.

Fixed-shape graphs per layer, sharing the same palettized weights:

  prefill (seq=S_PREFILL): x [1,S,3840] + causal/image mask [1,1,S,S]
      -> y [1,S,3840], k/v [1,8,S,256]  (post-RoPE, ready to cache)
  decode/verify (seq=Q, cache=S_MAX): x [1,Q,3840],
      mask [1,1,Q,S_MAX+Q], k_cache/v_cache [1,8,S_MAX,256]
      -> y [1,Q,3840], k_new/v_new [1,8,Q,256]

Q=1 emits the normal autoregressive decode bundle. Q>1 emits a speculative
verify bundle. The app passes [current token, draft_1, ..., draft_(Q-1)]; the
Q target logits verify the drafts and provide a correction/bonus token.

The app owns the cache buffers: after prefill it copies k/v into slots
[0..S), then commits the accepted prefix of k_new/v_new at the generation
slots. Attention runs over [cache ; query] = S_MAX+Q keys; invalid cache slots
and future query positions are masked by the app with -inf.

Constraint: S_MAX+Q must stay <= sliding_window (1024) so sliding and full
attention layers share the same mask semantics. num_kv_shared_layers is 0
for this checkpoint (verified), so every layer computes its own KV.

Validation (--validate, default on): for one sliding and one full layer,
prefill(4)+decode(Q) must match the cache-free forward over seq 4+Q.
"""

from __future__ import annotations

import argparse
import copy
import gc
import os
import time
from pathlib import Path

import coremltools as ct
import coremltools.optimize.coreml as cto
import numpy as np
import torch
from transformers import AutoModelForImageTextToText


class RecordCache:
    """Duck-typed HF Cache for prefill: captures post-RoPE K/V of the block
    and lets attention run over just that block."""

    def __init__(self):
        self.k = None
        self.v = None

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        self.k, self.v = key_states, value_states
        return key_states, value_states


class ConcatCache:
    """Duck-typed HF Cache for decode: attention runs over [cache ; new]."""

    def __init__(self, k_cache, v_cache):
        self.k_cache = k_cache
        self.v_cache = v_cache
        self.k = None
        self.v = None

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        self.k, self.v = key_states, value_states
        return (
            torch.cat([self.k_cache, key_states], dim=2),
            torch.cat([self.v_cache, value_states], dim=2),
        )


class PrefillLayerWrapper(torch.nn.Module):
    def __init__(self, language_model, layer_idx):
        super().__init__()
        self.rotary_emb = language_model.rotary_emb
        self.layer = language_model.layers[layer_idx]
        self.layer_type = language_model.config.layer_types[layer_idx]

    def forward(self, x, position_ids, attention_mask):
        cache = RecordCache()
        position_embeddings = self.rotary_emb(x, position_ids, self.layer_type)
        y = self.layer(
            x,
            shared_kv_states={},
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
        )
        return y, cache.k, cache.v


class DecodeLayerWrapper(torch.nn.Module):
    def __init__(self, language_model, layer_idx):
        super().__init__()
        self.rotary_emb = language_model.rotary_emb
        self.layer = language_model.layers[layer_idx]
        self.layer_type = language_model.config.layer_types[layer_idx]

    def forward(self, x, position_ids, attention_mask, k_cache, v_cache):
        cache = ConcatCache(k_cache, v_cache)
        position_embeddings = self.rotary_emb(x, position_ids, self.layer_type)
        y = self.layer(
            x,
            shared_kv_states={},
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
        )
        return y, cache.k, cache.v


class FullMaskWrapper(torch.nn.Module):
    """Cache-free reference (same math as the shipped seq320 layers)."""

    def __init__(self, language_model, layer_idx):
        super().__init__()
        self.rotary_emb = language_model.rotary_emb
        self.layer = language_model.layers[layer_idx]
        self.layer_type = language_model.config.layer_types[layer_idx]

    def forward(self, x, position_ids, attention_mask):
        position_embeddings = self.rotary_emb(x, position_ids, self.layer_type)
        return self.layer(
            x,
            shared_kv_states={},
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
        )


def causal_mask(seq, dtype, device):
    mask = torch.zeros(1, 1, seq, seq, dtype=dtype, device=device)
    mask.masked_fill_(torch.triu(torch.ones(seq, seq, dtype=torch.bool, device=device), diagonal=1), -65504.0)
    return mask


def cached_causal_mask(*, query_len, cache_len, written, dtype, device):
    """Mask [cache ; query] with a causal query tail.

    Only cache slots [0, written) are visible. Query row r sees query columns
    [0, r], which lets one target pass verify Q-1 speculative candidates and
    produce one correction/bonus token.
    """
    if not 0 <= written <= cache_len:
        raise ValueError(f"written must be within 0...{cache_len}, got {written}")
    mask = torch.full(
        (1, 1, query_len, cache_len + query_len),
        -65504.0,
        dtype=dtype,
        device=device,
    )
    mask[..., :written] = 0.0
    for row in range(query_len):
        mask[..., row, cache_len : cache_len + row + 1] = 0.0
    return mask


def layer_kv_shape(language_model, layer_idx):
    """KV cache geometry differs by layer type: sliding layers are GQA
    (8 heads x 256), full-attention layers are MQA with K=V projection
    (1 head x global_head_dim 512). Derive from the layer itself."""
    attn = language_model.layers[layer_idx].self_attn
    head_dim = attn.head_dim
    kv_heads = attn.k_proj.out_features // head_dim
    return kv_heads, head_dim


@torch.inference_mode()
def validate_layer(language_model, layer_idx, dtype, device, decode_seq):
    hidden = language_model.config.hidden_size
    prefill_len = 4
    seq = prefill_len + decode_seq
    x = (torch.randn(1, seq, hidden, device=device, dtype=dtype) * 0.5)
    pos = torch.arange(seq, device=device, dtype=torch.int32).unsqueeze(0)

    ref = FullMaskWrapper(language_model, layer_idx).eval()
    y_ref = ref(x, pos, causal_mask(seq, dtype, device))

    prefill = PrefillLayerWrapper(language_model, layer_idx).eval()
    decode = DecodeLayerWrapper(language_model, layer_idx).eval()

    kv_heads, head_dim = layer_kv_shape(language_model, layer_idx)
    s_max = seq  # small test cache
    k_cache = torch.zeros(1, kv_heads, s_max, head_dim, device=device, dtype=dtype)
    v_cache = torch.zeros_like(k_cache)

    y_pre, k_blk, v_blk = prefill(
        x[:, :prefill_len], pos[:, :prefill_len], causal_mask(prefill_len, dtype, device)
    )
    k_cache[:, :, :prefill_len] = k_blk
    v_cache[:, :, :prefill_len] = v_blk

    max_err = (y_pre - y_ref[:, :prefill_len]).abs().max().item()
    mask = cached_causal_mask(
        query_len=decode_seq,
        cache_len=s_max,
        written=prefill_len,
        dtype=dtype,
        device=device,
    )
    y_step, _, _ = decode(
        x[:, prefill_len:],
        pos[:, prefill_len:],
        mask,
        k_cache,
        v_cache,
    )
    max_err = max(max_err, (y_step - y_ref[:, prefill_len:]).abs().max().item())

    scale = y_ref.abs().max().item()
    rel = max_err / max(scale, 1e-6)
    print(f"validate layer{layer_idx:02d} type={prefill.layer_type} decode_seq={decode_seq} "
          f"maxAbsErr={max_err:.5f} refMax={scale:.2f} rel={rel:.5f}", flush=True)
    if rel > 5e-3:
        raise SystemExit(f"validation failed for layer {layer_idx}: rel {rel}")


def package_size(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for filename in files:
            total += os.path.getsize(os.path.join(root, filename))
    return total


def convert_one(*, name, wrapper, example_inputs, input_types, output_names,
                out_dir, quant_config, force):
    out_path = out_dir / f"{name}.mlpackage"
    if out_path.exists() and not force:
        print("skip_existing", out_path, flush=True)
        return
    print("convert", name, flush=True)
    # Trace on a CPU deep copy so `.cpu()` on the traced module never migrates
    # the *shared* language-model layer weights (each layer is converted twice —
    # prefill then decode — and the first conversion would otherwise leave the
    # layer on CPU, breaking the second with a cuda/cpu device mismatch).
    wrapper = copy.deepcopy(wrapper).cpu().eval()
    example_inputs = tuple(t.detach().cpu() for t in example_inputs)
    with torch.inference_mode():
        traced = torch.jit.trace(wrapper, example_inputs, strict=False)
    traced = traced.eval().cpu()
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=input_types,
        outputs=[ct.TensorType(name=n, dtype=np.float16) for n in output_names],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )
    qmodel = cto.palettize_weights(mlmodel, config=quant_config)
    qmodel.save(out_path)
    print("saved", out_path, "mib", round(package_size(out_path) / 1024**2, 2), flush=True)
    del qmodel, mlmodel, traced, wrapper
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/workspace/gemma12b/models/gemma-4-12b-it-qat-q4_0-unquantized")
    parser.add_argument("--out-dir", default="/workspace/gemma12b/coreml-layers-kv-pal4")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--prefill-seq", type=int, default=320)
    parser.add_argument("--s-max", type=int, default=512)
    parser.add_argument(
        "--decode-seq",
        type=int,
        default=1,
        help="query width; 1 creates normal decode bundles, >1 speculative verify bundles",
    )
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--target", choices=["all", "prefill", "decode"], default="all")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--skip-validate", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.decode_seq < 1:
        parser.error("--decode-seq must be at least 1")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("coremltools", ct.__version__, "torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_dir, dtype=torch.float16, device_map="cuda:0", low_cpu_mem_usage=True
    )
    model.eval()
    language_model = model.model.language_model
    config = language_model.config
    assert getattr(config, "num_kv_shared_layers", 0) == 0, "KV-shared layers not supported"
    assert args.s_max + args.decode_seq <= config.sliding_window, (
        "S_MAX + decode_seq must stay within the sliding window"
    )
    print("loaded_sec", round(time.time() - t0, 2), flush=True)

    device, dtype = "cuda", torch.float16
    hidden = config.hidden_size

    # Validate one sliding layer and one full-attention layer.
    full_idx = next(i for i, t in enumerate(config.layer_types) if t == "full_attention")
    validate_layer(language_model, 0, dtype, device, args.decode_seq)
    validate_layer(language_model, full_idx, dtype, device, args.decode_seq)
    if args.validate_only:
        print("validation ok", flush=True)
        return

    quant_config = cto.OptimizationConfig(
        global_config=cto.OpPalettizerConfig(
            nbits=4, mode="kmeans",
            granularity="per_grouped_channel", group_size=args.group_size,
        )
    )

    layer_indices = (
        list(range(config.num_hidden_layers)) if args.layers == "all"
        else [int(part) for part in args.layers.split(",")]
    )

    s_pre, s_max, s_decode = args.prefill_seq, args.s_max, args.decode_seq
    for layer_idx in layer_indices:
        if args.target in {"all", "prefill"}:
            x = torch.randn(1, s_pre, hidden, device=device, dtype=dtype) * 0.1
            pos = torch.arange(s_pre, device=device, dtype=torch.int32).unsqueeze(0)
            mask = causal_mask(s_pre, dtype, device)
            convert_one(
                name=f"gemma4_12b_layer{layer_idx:02d}_prefill_seq{s_pre}_kv_pal4_g{args.group_size}",
                wrapper=PrefillLayerWrapper(language_model, layer_idx),
                example_inputs=(x, pos, mask),
                input_types=[
                    ct.TensorType(name="x", shape=x.shape, dtype=np.float16),
                    ct.TensorType(name="position_ids", shape=pos.shape, dtype=np.int32),
                    ct.TensorType(name="attention_mask", shape=mask.shape, dtype=np.float16),
                ],
                output_names=["y", "k_block", "v_block"],
                out_dir=out_dir, quant_config=quant_config, force=args.force,
            )
        if args.target in {"all", "decode"}:
            kv_heads, head_dim = layer_kv_shape(language_model, layer_idx)
            x1 = torch.randn(1, s_decode, hidden, device=device, dtype=dtype) * 0.1
            pos1 = torch.arange(
                s_pre, s_pre + s_decode, device=device, dtype=torch.int32
            ).unsqueeze(0)
            mask1 = cached_causal_mask(
                query_len=s_decode,
                cache_len=s_max,
                written=s_pre,
                dtype=dtype,
                device=device,
            )
            kc = torch.randn(1, kv_heads, s_max, head_dim, device=device, dtype=dtype) * 0.1
            vc = torch.randn_like(kc) * 0.1
            if s_decode == 1:
                decode_name = f"gemma4_12b_layer{layer_idx:02d}_decode_kv{s_max}_pal4_g{args.group_size}"
            else:
                decode_name = (
                    f"gemma4_12b_layer{layer_idx:02d}_verify_seq{s_decode}_"
                    f"kv{s_max}_pal4_g{args.group_size}"
                )
            convert_one(
                name=decode_name,
                wrapper=DecodeLayerWrapper(language_model, layer_idx),
                example_inputs=(x1, pos1, mask1, kc, vc),
                input_types=[
                    ct.TensorType(name="x", shape=x1.shape, dtype=np.float16),
                    ct.TensorType(name="position_ids", shape=pos1.shape, dtype=np.int32),
                    ct.TensorType(name="attention_mask", shape=mask1.shape, dtype=np.float16),
                    ct.TensorType(name="k_cache", shape=kc.shape, dtype=np.float16),
                    ct.TensorType(name="v_cache", shape=vc.shape, dtype=np.float16),
                ],
                output_names=["y", "k_new", "v_new"],
                out_dir=out_dir, quant_config=quant_config, force=args.force,
            )

    print("done", flush=True)


if __name__ == "__main__":
    main()
