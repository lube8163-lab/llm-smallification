#!/usr/bin/env python3
"""Verify compiled Core ML bundles staged for the iOS probe."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


DEFAULT_MODEL_DIR = Path("ios/CoreMLProbe/CoreMLProbe/Models")
NORM_LM_HEAD = "gemma4_12b_norm_lm_head_1tok_int4_block32.mlmodelc"
LEGACY_LM_HEAD = "gemma4_12b_lm_head_1tok_int4_block32.mlmodelc"


def embedding_name(seq_len: int) -> str:
    return f"gemma4_12b_embedding_seq{seq_len}_int4_block32.mlmodelc"


def decoder_re(seq_len: int) -> re.Pattern[str]:
    return re.compile(
        rf"^gemma4_12b_layer(?P<index>\d+)_decoder_seq{seq_len}_mask_int4_block32\.mlmodelc$"
    )


def decoder_chunk_re(seq_len: int) -> re.Pattern[str]:
    return re.compile(
        rf"^gemma4_12b_layers(?P<start>\d+)_(?P<end>\d+)_decoder_seq{seq_len}_mask_int4_block32\.mlmodelc$"
    )


class Reporter:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def ok(self, message: str) -> None:
        print(f"OK: {message}")

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        print(f"WARN: {message}")

    def error(self, message: str) -> None:
        self.errors.append(message)
        print(f"ERROR: {message}")


def package_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def gib(size: int) -> str:
    return f"{size / 1024**3:.3f} GiB"


def read_mil(bundle: Path, reporter: Reporter) -> str:
    mil_path = bundle / "model.mil"
    if not mil_path.is_file():
        reporter.error(f"{bundle.name} is missing model.mil")
        return ""
    return mil_path.read_text(errors="replace")


def verify_required_bundle(model_dir: Path, name: str, reporter: Reporter) -> bool:
    path = model_dir / name
    if not path.is_dir():
        reporter.error(f"missing {name}")
        return False
    reporter.ok(f"{name} present ({gib(package_size(path))})")
    return True


def verify_decoders(model_dir: Path, expected_layers: int, seq_len: int, reporter: Reporter) -> None:
    by_index: dict[int, list[str]] = {}
    ignored: list[str] = []
    layer_pattern = decoder_re(seq_len)
    chunk_pattern = decoder_chunk_re(seq_len)

    for path in model_dir.glob(f"gemma4_12b_layer[0-9]*_decoder_seq{seq_len}_mask_int4_block32.mlmodelc"):
        match = layer_pattern.match(path.name)
        if not match:
            ignored.append(path.name)
            continue
        by_index.setdefault(int(match.group("index")), []).append(path.name)

    for path in model_dir.glob(f"gemma4_12b_layers[0-9]*_decoder_seq{seq_len}_mask_int4_block32.mlmodelc"):
        match = chunk_pattern.match(path.name)
        if not match:
            ignored.append(path.name)
            continue
        start = int(match.group("start"))
        end = int(match.group("end"))
        if end < start:
            reporter.error(f"invalid decoder chunk range: {path.name}")
            continue
        for index in range(start, end + 1):
            by_index.setdefault(index, []).append(path.name)

    if ignored:
        reporter.warn(f"ignored decoder-like names: {', '.join(sorted(ignored))}")

    expected = set(range(expected_layers))
    actual = set(by_index)
    missing = sorted(expected - actual)
    extra = sorted(index for index in actual if index not in expected)
    duplicates = {index: names for index, names in by_index.items() if len(names) > 1}

    if missing:
        reporter.error(f"missing decoder layer indices: {missing}")
    if extra:
        reporter.error(f"unexpected decoder layer indices: {extra}")
    if duplicates:
        details = "; ".join(
            f"{index}: {', '.join(sorted(names))}" for index, names in sorted(duplicates.items())
        )
        reporter.warn(f"duplicate decoder bundles: {details}")

    if not missing and not extra:
        bundle_count = len({name for names in by_index.values() for name in names})
        reporter.ok(
            f"decoder layers cover 0...{expected_layers - 1} "
            f"({len(actual)} layer indices, {bundle_count} bundle(s))"
        )


def verify_lm_head(
    model_dir: Path,
    *,
    require_norm_lm_head: bool,
    allow_legacy_lm_head: bool,
    fail_on_legacy: bool,
    reporter: Reporter,
) -> None:
    norm_path = model_dir / NORM_LM_HEAD
    legacy_path = model_dir / LEGACY_LM_HEAD
    has_norm = norm_path.is_dir()
    has_legacy = legacy_path.is_dir()

    if has_norm:
        reporter.ok(f"{NORM_LM_HEAD} present ({gib(package_size(norm_path))})")
        mil = read_mil(norm_path, reporter)
        if mil:
            if "tanh" in mil:
                reporter.ok("norm LM head MIL includes tanh final-logit softcap")
            else:
                reporter.error("norm LM head MIL does not include tanh final-logit softcap")

            norm_markers = ("rsqrt", "sqrt", "reduce", "rms", "norm")
            if any(marker in mil.lower() for marker in norm_markers):
                reporter.ok("norm LM head MIL has normalization-like ops")
            else:
                reporter.warn("could not find normalization-like ops in norm LM head MIL")
    elif require_norm_lm_head:
        reporter.error(f"missing preferred {NORM_LM_HEAD}")

    if has_legacy:
        mil = read_mil(legacy_path, reporter)
        if "tanh" not in mil:
            reporter.warn(f"{LEGACY_LM_HEAD} appears linear-only and lacks final softcap")
        if fail_on_legacy:
            reporter.error(f"legacy LM head still present: {LEGACY_LM_HEAD}")
        elif not has_norm and not allow_legacy_lm_head:
            reporter.error(
                f"only legacy LM head is present; rerun endpoint conversion or pass --allow-legacy-lm-head"
            )
        else:
            reporter.warn(f"legacy LM head present: {LEGACY_LM_HEAD}")

    if not has_norm and not has_legacy:
        reporter.error(f"missing {NORM_LM_HEAD} or {LEGACY_LM_HEAD}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", nargs="?", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--expected-layers", type=int, default=48)
    parser.add_argument("--seq-len", type=int, default=4)
    parser.add_argument("--require-norm-lm-head", action="store_true")
    parser.add_argument("--allow-legacy-lm-head", action="store_true")
    parser.add_argument("--fail-on-legacy", action="store_true")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    reporter = Reporter()

    if not model_dir.is_dir():
        reporter.error(f"missing model directory: {model_dir}")
    else:
        verify_required_bundle(model_dir, embedding_name(args.seq_len), reporter)
        verify_decoders(model_dir, args.expected_layers, args.seq_len, reporter)
        verify_lm_head(
            model_dir,
            require_norm_lm_head=args.require_norm_lm_head,
            allow_legacy_lm_head=args.allow_legacy_lm_head,
            fail_on_legacy=args.fail_on_legacy,
            reporter=reporter,
        )

    print(
        f"summary: {len(reporter.errors)} error(s), {len(reporter.warnings)} warning(s)"
    )
    return 1 if reporter.errors else 0


if __name__ == "__main__":
    sys.exit(main())
