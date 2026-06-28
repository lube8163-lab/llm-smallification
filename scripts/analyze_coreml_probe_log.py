#!/usr/bin/env python3
"""Summarize and sanity-check CoreMLProbe device logs."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


PREFERRED_LM_HEAD = "gemma4_12b_norm_lm_head_1tok_int4_block32"
LEGACY_LM_HEAD = "gemma4_12b_lm_head_1tok_int4_block32"


CRASH_MARKERS = (
    "EXC_RESOURCE",
    "RESOURCE_TYPE_MEMORY",
    "high watermark memory limit exceeded",
    "Fatal error",
    "Traceback ",
)

RUN_STARTED_RE = re.compile(r"\[CoreMLProbe\] run started (?P<detail>.*)")
RUN_FINISHED_RE = re.compile(r"\[CoreMLProbe\] run finished (?P<summary>.*)")
PEAK_MEMORY_RE = re.compile(r"\[CoreMLProbe\] Peak memory .*memory=(?P<memory>[0-9.]+) MB")
GENERATED_TOKENS_RE = re.compile(r"\[CoreMLProbe\] Generated tokens .*detail=(?P<detail>.*)")
GENERATED_TOKEN_RE = re.compile(r"\b(?P<index>\d+):#(?P<token>\d+)=(?P<logit>-?[0-9.]+)")
DECODER_STACK_RE = re.compile(r"\[CoreMLProbe\] Decoder stack .*detail=.*selected=(?P<selected>\d+)")


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


def read_log(path: str | None) -> str:
    if path is None or path == "-":
        return sys.stdin.read()
    return Path(path).read_text(errors="replace")


def parse_generated_tokens(detail: str) -> list[str]:
    return [match.group("token") for match in GENERATED_TOKEN_RE.finditer(detail)]


def longest_repeated_run(tokens: list[str]) -> tuple[str, int]:
    if not tokens:
        return "", 0

    best_token = tokens[0]
    best_count = 1
    current_token = tokens[0]
    current_count = 1

    for token in tokens[1:]:
        if token == current_token:
            current_count += 1
        else:
            current_token = token
            current_count = 1

        if current_count > best_count:
            best_token = current_token
            best_count = current_count

    return best_token, best_count


def first_match(pattern: re.Pattern[str], text: str) -> re.Match[str] | None:
    return pattern.search(text)


def analyze(text: str, args: argparse.Namespace) -> int:
    reporter = Reporter()
    coreml_lines = [line for line in text.splitlines() if "[CoreMLProbe]" in line]

    if not coreml_lines:
        reporter.error("no [CoreMLProbe] lines found")
    else:
        reporter.ok(f"found {len(coreml_lines)} CoreMLProbe log lines")

    run_started = first_match(RUN_STARTED_RE, text)
    if run_started:
        detail = run_started.group("detail")
        reporter.ok(f"run started: {detail}")
    else:
        reporter.warn("missing run started line")

    run_finished = first_match(RUN_FINISHED_RE, text)
    if run_finished:
        summary = run_finished.group("summary")
        if summary.startswith("OK"):
            reporter.ok(f"run finished: {summary}")
        else:
            reporter.error(f"run did not finish OK: {summary}")
    else:
        reporter.warn("missing run finished line")

    for marker in CRASH_MARKERS:
        if marker in text:
            reporter.error(f"crash/error marker found: {marker}")

    if "[CoreMLProbe] Error" in text:
        reporter.error("CoreMLProbe Error step found")

    has_target = "LM head target" in text or f"Load {PREFERRED_LM_HEAD}" in text
    has_preferred_load = f"Load {PREFERRED_LM_HEAD}" in text
    has_fallback = "LM head fallback" in text or f"Load {LEGACY_LM_HEAD}" in text

    if has_target:
        reporter.ok("preferred norm+lm_head endpoint was selected")
    elif args.require_norm_lm_head:
        reporter.error(f"preferred endpoint was not selected: {PREFERRED_LM_HEAD}")
    else:
        reporter.warn(f"preferred endpoint not observed: {PREFERRED_LM_HEAD}")

    if has_preferred_load:
        reporter.ok(f"preferred endpoint loaded: {PREFERRED_LM_HEAD}")
    elif args.require_norm_lm_head:
        reporter.error(f"preferred endpoint load not observed: {PREFERRED_LM_HEAD}")

    if has_fallback:
        message = f"legacy/fallback LM head observed: {LEGACY_LM_HEAD}"
        if args.fail_on_fallback or args.require_norm_lm_head:
            reporter.error(message)
        else:
            reporter.warn(message)
    else:
        reporter.ok("no LM head fallback observed")

    decoder_matches = list(DECODER_STACK_RE.finditer(text))
    if decoder_matches:
        selected = int(decoder_matches[-1].group("selected"))
        if args.expect_layers is not None and selected != args.expect_layers:
            reporter.error(f"decoder selected {selected} layers, expected {args.expect_layers}")
        else:
            reporter.ok(f"decoder selected {selected} layers")
    elif args.expect_layers is not None:
        reporter.error("decoder stack selection not found")

    peak_match = first_match(PEAK_MEMORY_RE, text)
    if peak_match:
        peak = float(peak_match.group("memory"))
        if args.max_peak_mb is not None and peak > args.max_peak_mb:
            reporter.error(f"peak memory {peak:.1f} MB exceeds {args.max_peak_mb:.1f} MB")
        else:
            reporter.ok(f"peak memory {peak:.1f} MB")
    elif args.max_peak_mb is not None:
        reporter.error("peak memory not found")
    else:
        reporter.warn("peak memory not found")

    generated_match = first_match(GENERATED_TOKENS_RE, text)
    if generated_match:
        tokens = parse_generated_tokens(generated_match.group("detail"))
        unique_tokens = sorted(set(tokens))
        reporter.ok(f"generated {len(tokens)} token(s): {','.join('#' + token for token in tokens)}")

        if args.min_generated_tokens is not None and len(tokens) < args.min_generated_tokens:
            reporter.error(
                f"generated {len(tokens)} token(s), expected at least {args.min_generated_tokens}"
            )

        if len(tokens) > 1 and len(unique_tokens) == 1:
            message = f"all generated tokens are identical: #{unique_tokens[0]}"
            if args.fail_on_repeat:
                reporter.error(message)
            else:
                reporter.warn(message)

        repeat_token, repeat_count = longest_repeated_run(tokens)
        if repeat_count > args.max_repeat_run:
            message = (
                f"generated token #{repeat_token} repeated {repeat_count} consecutive times "
                f"(allowed {args.max_repeat_run})"
            )
            if args.fail_on_repeat:
                reporter.error(message)
            else:
                reporter.warn(message)
    elif args.min_generated_tokens is not None:
        reporter.error("generated token summary not found")
    else:
        reporter.warn("generated token summary not found")

    repeated_input_count = text.count("Repeated input window")
    if repeated_input_count:
        message = f"repeated input window observed {repeated_input_count} time(s)"
        if args.fail_on_repeat:
            reporter.error(message)
        else:
            reporter.warn(message)

    print(f"summary: {len(reporter.errors)} error(s), {len(reporter.warnings)} warning(s)")
    return 1 if reporter.errors else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", nargs="?", help="log file path, or stdin when omitted")
    parser.add_argument("--require-norm-lm-head", action="store_true")
    parser.add_argument("--fail-on-fallback", action="store_true")
    parser.add_argument("--fail-on-repeat", action="store_true")
    parser.add_argument("--expect-layers", type=int)
    parser.add_argument("--min-generated-tokens", type=int)
    parser.add_argument("--max-peak-mb", type=float)
    parser.add_argument("--max-repeat-run", type=int, default=3)
    args = parser.parse_args()

    return analyze(read_log(args.log), args)


if __name__ == "__main__":
    sys.exit(main())
