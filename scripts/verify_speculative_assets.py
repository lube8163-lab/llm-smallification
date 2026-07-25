#!/usr/bin/env python3
"""Verify the compiled Core ML assets required by speculative KV decoding."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


DEFAULT_MODEL_DIR = Path("ios/CoreMLProbe/CoreMLProbe/Models")
DEFAULT_DRAFTER_NAMES = (
    "gemma4_12b_drafter_step_kv512_mixed_pal4_g16_head_int8.mlmodelc",
)


def bundle_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def verify_name(layer: int, verify_seq: int, cache_size: int) -> str:
    return (
        f"gemma4_12b_layer{layer:02d}_verify_seq{verify_seq}_"
        f"kv{cache_size}_pal4_g16.mlmodelc"
    )


def read_mil(bundle: Path) -> str:
    path = bundle / "model.mil"
    return path.read_text(errors="replace") if path.is_file() else ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", nargs="?", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--expected-layers", type=int, default=48)
    parser.add_argument("--verify-seq", type=int, default=4)
    parser.add_argument("--cache-size", type=int, default=512)
    parser.add_argument("--drafter-name")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    errors: list[str] = []
    if not model_dir.is_dir():
        print(f"ERROR: missing model directory: {model_dir}")
        return 1

    drafter_candidates = (
        (f"{args.drafter_name}.mlmodelc",)
        if args.drafter_name and not args.drafter_name.endswith(".mlmodelc")
        else ((args.drafter_name,) if args.drafter_name else DEFAULT_DRAFTER_NAMES)
    )
    drafter = next(
        (model_dir / name for name in drafter_candidates if (model_dir / name).is_dir()),
        None,
    )
    if drafter is None:
        errors.append(f"missing drafter; expected one of {', '.join(drafter_candidates)}")
    else:
        print(f"OK: drafter {drafter.name} ({bundle_size(drafter) / 1024**2:.1f} MiB)")
        mil = read_mil(drafter)
        for marker in (
            "backbone_hidden",
            "token_emb",
            "sliding_k",
            "sliding_v",
            "full_k",
            "full_v",
            "position_ids",
            "attention_mask",
            "logits",
            "projected_hidden",
        ):
            if mil and marker not in mil:
                errors.append(f"{drafter.name} MIL is missing {marker}")

    missing_layers: list[int] = []
    verify_bundles: list[Path] = []
    for layer in range(args.expected_layers):
        bundle = model_dir / verify_name(layer, args.verify_seq, args.cache_size)
        if not bundle.is_dir():
            missing_layers.append(layer)
        else:
            verify_bundles.append(bundle)
    if missing_layers:
        errors.append(f"missing verify layer indices: {missing_layers}")
    else:
        total = sum(bundle_size(bundle) for bundle in verify_bundles)
        print(
            f"OK: verify layers cover 0...{args.expected_layers - 1} "
            f"({len(verify_bundles)} bundles, {total / 1024**3:.3f} GiB)"
        )

    for bundle in (verify_bundles[:1] + verify_bundles[-1:]):
        mil = read_mil(bundle)
        for marker in ("position_ids", "attention_mask", "k_cache", "v_cache", "k_new", "v_new"):
            if mil and marker not in mil:
                errors.append(f"{bundle.name} MIL is missing {marker}")

    for error in errors:
        print(f"ERROR: {error}")
    print(f"summary: {len(errors)} error(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
