#!/usr/bin/env python3
"""Export the Gemma tokenizer JSON into the CoreMLProbe resource folder."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_MODEL = "google/gemma-4-12B-it-qat-q4_0-unquantized"
DEFAULT_CACHE = Path("runpod-artifacts/tokenizer/gemma4-tokenizer.json")
DEFAULT_DST = Path("ios/CoreMLProbe/CoreMLProbe/Models/gemma4-tokenizer.json")


def export_tokenizer(model: str, output: Path) -> None:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: pip install transformers sentencepiece jinja2"
        ) from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    tokenizer.backend_tokenizer.save(str(output))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.force or not args.cache.is_file():
        export_tokenizer(args.model, args.cache)

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.cache, args.dst)
    print(f"copied tokenizer: {args.dst} ({args.dst.stat().st_size / 1024**2:.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
