#!/usr/bin/env python3
"""Host-side Gemma 4 tokenizer helper for the fixed-window Core ML probe."""

from __future__ import annotations

import argparse
from typing import Iterable


DEFAULT_MODEL = "google/gemma-4-12B-it-qat-q4_0-unquantized"


def parse_token_ids(value: str) -> list[int]:
    result: list[int] = []
    for item in value.replace(" ", ",").split(","):
        item = item.strip()
        if not item:
            continue
        result.append(int(item.removeprefix("#")))
    return result


def format_token_ids(values: Iterable[int]) -> str:
    return ",".join(str(value) for value in values)


def load_tokenizer(model: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: pip install transformers sentencepiece jinja2"
        ) from exc

    return AutoTokenizer.from_pretrained(model, trust_remote_code=True)


def extract_input_ids(value) -> list[int]:
    if isinstance(value, list):
        if value and isinstance(value[0], list):
            return [int(item) for item in value[0]]
        return [int(item) for item in value]

    if hasattr(value, "input_ids"):
        return extract_input_ids(value.input_ids)

    if isinstance(value, dict) and "input_ids" in value:
        return extract_input_ids(value["input_ids"])

    raise TypeError(f"unexpected tokenizer output: {type(value).__name__}")


def encode_prompt(tokenizer, prompt: str, use_chat_template: bool) -> list[int]:
    if use_chat_template:
        messages = [{"role": "user", "content": prompt}]
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        return extract_input_ids(encoded)
    return tokenizer.encode(prompt, add_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt")
    parser.add_argument("--decode-ids")
    parser.add_argument(
        "--window-sizes",
        default="4,20",
        help="comma-separated trailing windows to print from input_ids",
    )
    parser.add_argument(
        "--chat-template",
        action="store_true",
        help="include the model chat template; this is the default unless --raw is passed",
    )
    parser.add_argument("--raw", action="store_true", help="tokenize the prompt without the chat template")
    args = parser.parse_args()

    if not args.prompt and not args.decode_ids:
        parser.error("pass --prompt, --decode-ids, or both")

    tokenizer = load_tokenizer(args.model)

    if args.prompt:
        ids = encode_prompt(
            tokenizer,
            args.prompt,
            use_chat_template=not args.raw,
        )
        print("input_ids=" + format_token_ids(ids))
        for raw_size in args.window_sizes.split(","):
            raw_size = raw_size.strip()
            if not raw_size:
                continue
            size = int(raw_size)
            window = ids[-size:]
            print(f"input_ids_last{size}=" + format_token_ids(window))

    if args.decode_ids:
        ids = parse_token_ids(args.decode_ids)
        print("decoded=" + tokenizer.decode(ids, skip_special_tokens=False))


if __name__ == "__main__":
    main()
