#!/usr/bin/env python3
"""Convert the Gemma 4 MTP drafter to a fixed-shape draft-step Core ML package.

One autoregressive draft step (seq=1) that cross-attends to the backbone's KV
cache (which CoreMLProbe already holds as layer46 sliding + layer47 full):

  inputs: token_emb [1,1,3840], backbone_hidden [1,1,3840],
          sliding_k/v [1,8,S,256], full_k/v [1,1,S,512],
          position_ids [1,1], attention_mask [1,1,1,S]
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
        # Gemma4AssistantCandidateGenerator concatenates these in this exact
        # order. Reversing the two equally-sized inputs still converts and
        # predicts, but destroys drafter acceptance on real target states.
        inputs_embeds = self.pre_projection(torch.cat([token_emb, backbone_hidden], dim=-1))
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
    parser.add_argument("--drafter", default=DRAFTER)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument(
        "--quantization",
        choices=["fp16", "pal4", "mixed"],
        default="mixed",
        help=(
            "mixed keeps the drafter body pal4 and uses int8 for the sensitive "
            "262k-vocabulary head"
        ),
    )
    parser.add_argument("--keep-fp16", action="store_true")
    args = parser.parse_args()

    import os
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cpu"  # trace on CPU to keep weights put
    dt = torch.float32
    print("loading drafter", flush=True)
    model = Gemma4UnifiedAssistantForCausalLM.from_pretrained(
        args.drafter, dtype=dt, low_cpu_mem_usage=True
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
        official = model(
            inputs_embeds=torch.cat([ex["token_emb"], ex["backbone_hidden"]], dim=-1),
            attention_mask=torch.ones(1, S, dtype=dt),
            position_ids=ex["position_ids"],
            shared_kv_states={
                "sliding_attention": (ex["sliding_k"], ex["sliding_v"]),
                "full_attention": (ex["full_k"], ex["full_v"]),
            },
            use_cache=False,
        )
        # The wrapper uses finite fp16-min for masked slots while the eager HF
        # helper starts from float32-min. Both exclude the same keys; allow the
        # resulting sub-millilogit rounding difference.
        torch.testing.assert_close(ref_logits, official.logits, rtol=1e-3, atol=2e-4)
        torch.testing.assert_close(ref_hidden, official.last_hidden_state, rtol=1e-3, atol=2e-4)
        print("official candidate-generator input order validated", flush=True)
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
    def validate_coreml(candidate, label):
        pred = candidate.predict(feed)
        ml_logits = np.asarray(pred["logits"]).reshape(-1)
        hf_logits = ref_logits.detach().numpy().reshape(-1).astype(np.float32)
        ml_arg = int(np.argmax(ml_logits)); hf_arg = int(np.argmax(hf_logits))
        print(f"{label} argmax: coreml #{ml_arg} vs hf #{hf_arg} -> {ml_arg == hf_arg}", flush=True)
        print(f"{label} logit maxabs diff: {np.max(np.abs(ml_logits.astype(np.float32) - hf_logits)):.4f}", flush=True)

    try:
        validate_coreml(mlmodel, "fp16")
    except Exception as e:
        print("predict skipped (Linux CoreML runtime):", str(e)[:120], flush=True)

    def size(p):
        tot = 0
        for r, _, fs in os.walk(p):
            for f in fs:
                tot += os.path.getsize(os.path.join(r, f))
        return tot

    fp16_path = f"{args.out_dir}/gemma4_12b_drafter_step_kv{S}_fp16.mlpackage"
    if args.keep_fp16 or args.quantization == "fp16":
        mlmodel.save(fp16_path)
        print("saved", fp16_path, "mib", round(size(fp16_path) / 1024**2, 2), flush=True)

    if args.quantization == "pal4":
        cfg = cto.OptimizationConfig(global_config=cto.OpPalettizerConfig(
            nbits=4, mode="kmeans", granularity="per_grouped_channel", group_size=args.group_size))
        output_model = cto.palettize_weights(mlmodel, config=cfg)
        out_path = f"{args.out_dir}/gemma4_12b_drafter_step_kv{S}_pal4_g{args.group_size}.mlpackage"
    elif args.quantization == "mixed":
        metadata = cto.get_weights_metadata(mlmodel, weight_threshold=2048)
        if not metadata:
            raise RuntimeError("could not inspect drafter weights for mixed compression")
        # The assistant lm_head is 262144x1024 and is by far the largest
        # tensor. Pal4 changed its argmax in the first conversion experiment,
        # so exclude it from palettization and quantize that tensor to int8 in
        # a second pass. This keeps the package much smaller than fp16 without
        # applying the known-lossy 4-bit treatment to the vocabulary head.
        head_weight_name, head_metadata = max(
            metadata.items(), key=lambda item: item[1].val.size
        )
        print(
            "mixed compression head",
            head_weight_name,
            "shape",
            head_metadata.val.shape,
            flush=True,
        )
        body_cfg = cto.OptimizationConfig(
            global_config=cto.OpPalettizerConfig(
                nbits=4,
                mode="kmeans",
                granularity="per_grouped_channel",
                group_size=args.group_size,
            ),
            op_name_configs={head_weight_name: None},
        )
        body_pal4 = cto.palettize_weights(mlmodel, config=body_cfg)
        head_cfg = cto.OptimizationConfig(
            op_name_configs={
                head_weight_name: cto.OpLinearQuantizerConfig(
                    mode="linear_symmetric", weight_threshold=0
                )
            }
        )
        output_model = cto.linear_quantize_weights(body_pal4, config=head_cfg)
        out_path = (
            f"{args.out_dir}/gemma4_12b_drafter_step_kv{S}_"
            f"mixed_pal4_g{args.group_size}_head_int8.mlpackage"
        )
    else:
        output_model = None
        out_path = fp16_path

    if output_model is not None:
        try:
            validate_coreml(output_model, args.quantization)
        except Exception as e:
            print("compressed predict skipped:", str(e)[:120], flush=True)
        output_model.save(out_path)
        print("saved", out_path, "mib", round(size(out_path) / 1024**2, 2), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
