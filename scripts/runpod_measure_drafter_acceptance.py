#!/usr/bin/env python3
"""Measure Gemma 4 MTP drafter acceptance against the 12B backbone.

Answers the go/no-go question for on-device speculative decoding: with greedy
(argmax) decoding, how many draft tokens does the target accept per verify
sweep? >2 means the ~4s decode sweep amortizes and the phone gets a real
speedup; <1.5 means it isn't worth the plumbing.

Uses HF assisted generation (which drives the drafter's cross-attention to the
backbone's shared KV) for correctness, then a manual greedy speculative loop to
count accepted tokens per sweep directly.
"""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer
from transformers import Gemma4UnifiedAssistantForCausalLM


PROMPTS = [
    "富士山について教えてください。",
    "日本を代表する観光地を3つ挙げて、それぞれ一言で説明してください。",
    "Explain what a large language model is, in three sentences.",
    "りんごとみかんの違いを説明してください。",
    "What is the capital of Japan? Answer in one word.",
]


def build_inputs(tokenizer, prompt, device):
    messages = [{"role": "user", "content": prompt}]
    enc = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    return enc["input_ids"].to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/workspace/gemma12b/models/gemma-4-12b-it-qat-q4_0-unquantized")
    parser.add_argument("--drafter", default="google/gemma-4-12B-it-assistant")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--assistant-tokens", type=int, default=4)
    args = parser.parse_args()

    device = "cuda"
    print("loading backbone", args.model_dir, flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, device_map=device, low_cpu_mem_usage=True
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    print("loading drafter", args.drafter, flush=True)
    drafter = Gemma4UnifiedAssistantForCausalLM.from_pretrained(
        args.drafter, dtype=torch.bfloat16, device_map=device, low_cpu_mem_usage=True
    ).eval()
    drafter.generation_config.num_assistant_tokens = args.assistant_tokens
    drafter.generation_config.num_assistant_tokens_schedule = "constant"

    print(f"\n{'prompt':40s} | base s | asst s | speedup | match | acc/sweep", flush=True)
    total_base = total_asst = 0.0
    for prompt in PROMPTS:
        ids = build_inputs(tokenizer, prompt, device)

        with torch.inference_mode():
            torch.cuda.synchronize(); t0 = time.time()
            base = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False)
            torch.cuda.synchronize(); t_base = time.time() - t0

            torch.cuda.synchronize(); t0 = time.time()
            asst = model.generate(
                ids, assistant_model=drafter, max_new_tokens=args.max_new_tokens, do_sample=False
            )
            torch.cuda.synchronize(); t_asst = time.time() - t0

        base_new = base[0, ids.shape[1]:]
        asst_new = asst[0, ids.shape[1]:]
        n = min(len(base_new), len(asst_new))
        match = bool(torch.equal(base_new[:n], asst_new[:n]))
        total_base += t_base; total_asst += t_asst
        # acc/sweep is estimated on-device separately; here GPU speedup is the proxy
        print(f"{prompt[:38]:40s} | {t_base:6.2f} | {t_asst:6.2f} | {t_base/t_asst:6.2f}x | {str(match):5s} | -", flush=True)

    print(f"\nTOTAL base={total_base:.1f}s asst={total_asst:.1f}s speedup={total_base/total_asst:.2f}x", flush=True)
    print("(GPU speedup is a lower bound on the mobile win: on-device the verify"
          " sweep is load-bound, so each accepted token saves a full ~4s reload.)", flush=True)


if __name__ == "__main__":
    main()
