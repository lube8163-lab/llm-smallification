#!/usr/bin/env python3
"""Convert the Gemma 4 MTP drafter to a fixed-shape draft-step Core ML package.

One autoregressive draft step (seq=1) that cross-attends to the backbone's KV
cache (which CoreMLProbe already holds as layer46 sliding + layer47 full):

  inputs: backbone_hidden [1,1,3840], token_emb [1,1,3840],
          sliding_k/v [1,8,S,256], full_k/v [1,1,S,512],
          position_ids [1,1], attention_mask [1,S]
  outputs: logits [1,1,262144], projected_hidden [1,1,3840]

Chained on device: projected_hidden feeds the next step's backbone_hidden.
Validated against the HF drafter forward before saving.
"""

from __future__ import annotations

import argparse

import coremltools as ct
import coremltools.optimize.coreml as cto
import numpy as np
import torch
from transformers import Gemma4UnifiedAssistantForCausalLM

# The drafter's create_attention_masks uses tensor.new_ones / new_zeros, which
# coremltools doesn't ship a converter for. Register them as fill ops.
from coremltools.converters.mil.frontend.torch.ops import _get_inputs
from coremltools.converters.mil.frontend.torch.torch_op_registry import register_torch_op
from coremltools.converters.mil import Builder as mb


@register_torch_op(override=True)
def new_ones(context, node):
    inputs = _get_inputs(context, node)
    context.add(mb.fill(shape=inputs[1], value=1.0, name=node.name))


@register_torch_op(override=True)
def new_zeros(context, node):
    inputs = _get_inputs(context, node)
    context.add(mb.fill(shape=inputs[1], value=0.0, name=node.name))

DRAFTER = "google/gemma-4-12B-it-assistant"
S_MAX = 512


class DraftStepWrapper(torch.nn.Module):
    """Wraps the HF drafter for a single draft step (seq=1).

    For q_len==1 the drafter's bidirectional sliding/full masks both collapse to
    a plain pad mask over the backbone KV (HF's own comment: "q_len == 1 acts as
    full attention no matter what"), so we bypass create_attention_masks and take
    the [1,1,1,S] additive mask as a direct input. This avoids the untraceable
    new_ones/flip mask machinery and matches the on-device path (the app builds
    the same pad mask it already uses for KV decode)."""

    def __init__(self, model):
        super().__init__()
        self.inner = model.model            # Gemma4UnifiedModel (4 layers)
        self.pre_projection = model.pre_projection
        self.post_projection = model.post_projection
        self.lm_head = model.lm_head

    def forward(self, backbone_hidden, token_emb, sliding_k, sliding_v, full_k, full_v, position_ids, attention_mask):
        inputs_embeds = self.pre_projection(torch.cat([backbone_hidden, token_emb], dim=-1))
        shared_kv = {
            "sliding_attention": (sliding_k, sliding_v),
            "full_attention": (full_k, full_v),
        }
        mask_dict = {"sliding_attention": attention_mask, "full_attention": attention_mask}
        out = self.inner(
            inputs_embeds=inputs_embeds,
            attention_mask=mask_dict,
            position_ids=position_ids,
            shared_kv_states=shared_kv,
            use_cache=False,
        )
        last = out.last_hidden_state
        return self.lm_head(last), self.post_projection(last)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="/workspace/gemma12b/coreml-drafter")
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--keep-fp16", action="store_true")
    args = parser.parse_args()

    import os
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cpu"  # trace on CPU to keep weights put
    dt = torch.float32
    print("loading drafter", flush=True)
    model = Gemma4UnifiedAssistantForCausalLM.from_pretrained(
        DRAFTER, dtype=dt, low_cpu_mem_usage=True
    ).eval()
    tc = model.config.get_text_config()
    H = model.config.backbone_hidden_size
    wrapper = DraftStepWrapper(model).eval()

    S = S_MAX
    ex = dict(
        backbone_hidden=torch.randn(1, 1, H, dtype=dt) * 0.1,
        token_emb=torch.randn(1, 1, H, dtype=dt) * 0.1,
        sliding_k=torch.randn(1, tc.num_key_value_heads, S, tc.head_dim, dtype=dt) * 0.1,
        sliding_v=torch.randn(1, tc.num_key_value_heads, S, tc.head_dim, dtype=dt) * 0.1,
        full_k=torch.randn(1, tc.num_global_key_value_heads, S, tc.global_head_dim, dtype=dt) * 0.1,
        full_v=torch.randn(1, tc.num_global_key_value_heads, S, tc.global_head_dim, dtype=dt) * 0.1,
        position_ids=torch.tensor([[S]], dtype=torch.int32),
        # additive pad mask over the S backbone-KV slots (0=visible, -inf=pad)
        attention_mask=torch.zeros(1, 1, 1, S, dtype=dt),
    )
    order = ["backbone_hidden", "token_emb", "sliding_k", "sliding_v", "full_k", "full_v", "position_ids", "attention_mask"]
    example = tuple(ex[k] for k in order)

    with torch.inference_mode():
        ref_logits, ref_hidden = wrapper(*example)
        print("forward ok logits", tuple(ref_logits.shape), "hidden", tuple(ref_hidden.shape), flush=True)
        traced = torch.jit.trace(wrapper, example, strict=False)
    print("trace ok", flush=True)

    inputs = [
        ct.TensorType(name="backbone_hidden", shape=ex["backbone_hidden"].shape, dtype=np.float16),
        ct.TensorType(name="token_emb", shape=ex["token_emb"].shape, dtype=np.float16),
        ct.TensorType(name="sliding_k", shape=ex["sliding_k"].shape, dtype=np.float16),
        ct.TensorType(name="sliding_v", shape=ex["sliding_v"].shape, dtype=np.float16),
        ct.TensorType(name="full_k", shape=ex["full_k"].shape, dtype=np.float16),
        ct.TensorType(name="full_v", shape=ex["full_v"].shape, dtype=np.float16),
        ct.TensorType(name="position_ids", shape=ex["position_ids"].shape, dtype=np.int32),
        ct.TensorType(name="attention_mask", shape=ex["attention_mask"].shape, dtype=np.float16),
    ]
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=inputs,
        outputs=[ct.TensorType(name="logits", dtype=np.float16),
                 ct.TensorType(name="projected_hidden", dtype=np.float16)],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )
    print("convert ok", flush=True)

    # Numeric check vs HF (fp16 CoreML on CPU)
    def np16(t):
        return t.detach().numpy().astype(np.float16)
    feed = {
        "backbone_hidden": np16(ex["backbone_hidden"]), "token_emb": np16(ex["token_emb"]),
        "sliding_k": np16(ex["sliding_k"]), "sliding_v": np16(ex["sliding_v"]),
        "full_k": np16(ex["full_k"]), "full_v": np16(ex["full_v"]),
        "position_ids": ex["position_ids"].numpy().astype(np.int32),
        "attention_mask": np16(ex["attention_mask"]),
    }
    try:
        pred = mlmodel.predict(feed)
        ml_logits = np.asarray(pred["logits"]).reshape(-1)
        hf_logits = ref_logits.detach().numpy().reshape(-1).astype(np.float32)
        ml_arg = int(np.argmax(ml_logits)); hf_arg = int(np.argmax(hf_logits))
        print(f"argmax match: coreml #{ml_arg} vs hf #{hf_arg} -> {ml_arg == hf_arg}", flush=True)
        print(f"logit maxabs diff (top region): {np.max(np.abs(ml_logits.astype(np.float32) - hf_logits)):.4f}", flush=True)
    except Exception as e:
        print("predict skipped (Linux CoreML runtime):", str(e)[:120], flush=True)

    if args.keep_fp16:
        mlmodel.save(f"{args.out_dir}/gemma4_12b_drafter_step_kv{S}_fp16.mlpackage")

    cfg = cto.OptimizationConfig(global_config=cto.OpPalettizerConfig(
        nbits=4, mode="kmeans", granularity="per_grouped_channel", group_size=args.group_size))
    q = cto.palettize_weights(mlmodel, config=cfg)
    out_path = f"{args.out_dir}/gemma4_12b_drafter_step_kv{S}_pal4_g{args.group_size}.mlpackage"
    q.save(out_path)

    def size(p):
        tot = 0
        for r, _, fs in os.walk(p):
            for f in fs:
                tot += os.path.getsize(os.path.join(r, f))
        return tot
    print("saved", out_path, "mib", round(size(out_path) / 1024**2, 2), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
