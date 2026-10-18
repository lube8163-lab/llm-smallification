#!/usr/bin/env python3
"""Probe the Gemma 4 MTP drafter's forward I/O to design the Core ML wrapper.

The drafter cross-attends to the backbone's shared KV (no own k/v proj), so we
need to know exact tensor shapes/dtypes it expects for shared_kv_states,
position_ids, attention_mask, and inputs_embeds before tracing it fixed-shape.
"""

from __future__ import annotations

import torch
from transformers import Gemma4UnifiedAssistantForCausalLM


DRAFTER = "google/gemma-4-12B-it-assistant"


def main():
    device = "cuda"
    m = Gemma4UnifiedAssistantForCausalLM.from_pretrained(
        DRAFTER, dtype=torch.bfloat16, device_map=device, low_cpu_mem_usage=True
    ).eval()
    tc = m.config.get_text_config()
    print("=== drafter config ===")
    print("hidden", tc.hidden_size, "layers", tc.num_hidden_layers,
          "layer_types", tc.layer_types, "kv_shared", tc.num_kv_shared_layers,
          "heads", tc.num_attention_heads, "kv_heads", tc.num_key_value_heads,
          "head_dim", tc.head_dim, "global_head_dim", tc.global_head_dim,
          "num_global_kv", tc.num_global_key_value_heads, flush=True)
    print("backbone_hidden", m.config.backbone_hidden_size,
          "use_ordered_emb", m.config.use_ordered_embeddings, flush=True)

    print("=== attention submodule of layer 0 (which projections exist?) ===")
    a0 = m.model.layers[0].self_attn
    print([n for n, _ in a0.named_parameters()], flush=True)
    a3 = m.model.layers[3].self_attn
    print("layer3 (full):", [n for n, _ in a3.named_parameters()], flush=True)

    # Build a minimal forward: K query tokens, backbone KV of length S.
    B, K, S = 1, 4, 40
    H = m.config.backbone_hidden_size
    dt = torch.bfloat16
    # inputs_embeds is the drafter's own input BEFORE pre_projection? In modeling,
    # forward() calls self.pre_projection(inputs_embeds) with inputs_embeds of
    # width 2*backbone_hidden. Confirm width.
    print("pre_projection.in_features", m.pre_projection.in_features,
          "out", m.pre_projection.out_features, flush=True)
    print("post_projection", m.post_projection.in_features, "->", m.post_projection.out_features, flush=True)

    inputs_embeds = torch.randn(B, K, m.pre_projection.in_features, device=device, dtype=dt) * 0.1
    kv_sliding = (
        torch.randn(B, tc.num_key_value_heads, S, tc.head_dim, device=device, dtype=dt) * 0.1,
        torch.randn(B, tc.num_key_value_heads, S, tc.head_dim, device=device, dtype=dt) * 0.1,
    )
    kv_full = (
        torch.randn(B, tc.num_global_key_value_heads, S, tc.global_head_dim, device=device, dtype=dt) * 0.1,
        torch.randn(B, tc.num_global_key_value_heads, S, tc.global_head_dim, device=device, dtype=dt) * 0.1,
    )
    shared_kv = {"sliding_attention": kv_sliding, "full_attention": kv_full}
    position_ids = torch.arange(S, S + K, device=device).unsqueeze(0)
    attn_mask = torch.ones(B, S, device=device, dtype=torch.long)

    with torch.inference_mode():
        out = m(
            inputs_embeds=inputs_embeds,
            shared_kv_states=shared_kv,
            position_ids=position_ids,
            attention_mask=attn_mask,
        )
    print("=== forward OK ===")
    print("logits", tuple(out.logits.shape), out.logits.dtype, flush=True)
    print("last_hidden_state (post_projected)", tuple(out.last_hidden_state.shape), flush=True)


if __name__ == "__main__":
    main()
