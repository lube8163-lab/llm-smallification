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
RUN_FAILED_RE = re.compile(r"\[CoreMLProbe\] run failed: (?P<summary>.*)")
PEAK_MEMORY_RE = re.compile(r"\[CoreMLProbe\] Peak memory .*memory=(?P<memory>[0-9.]+) MB")
GENERATED_TOKENS_RE = re.compile(r"\[CoreMLProbe\] Generated tokens .*detail=(?P<detail>.*)")
GENERATED_TOKEN_RE = re.compile(r"\b(?P<index>\d+):#(?P<token>\d+)=(?P<logit>-?[0-9.]+)")
DECODER_STACK_RE = re.compile(r"\[CoreMLProbe\] Decoder stack .*detail=.*selected=(?P<selected>\d+)")
GENERATION_LOOP_RE = re.compile(
    r"\[CoreMLProbe\] Generation loop .*detail=tokens=(?P<tokens>\d+).*retain decoders=(?P<retain>\d+)"
)
TOKEN_TOTAL_RE = re.compile(
    r"\[CoreMLProbe\] Token (?P<index>\d+) total duration=(?P<seconds>[0-9.]+)s memory=(?P<memory>[0-9.]+) MB"
)
PROMPT_IDS_RE = re.compile(r"\[CoreMLProbe\] Prompt IDs (?P<index>\d+)")
EMBEDDING_TOKEN_RE = re.compile(
    r"\[CoreMLProbe\] Embedding token (?P<index>\d+) duration=(?P<seconds>[0-9.]+)s"
)
DECODER_LOAD_RE = re.compile(
    r"\[CoreMLProbe\] Load gemma4_12b_layers\d+_\d+_decoder_.* duration=(?P<seconds>[0-9.]+)s"
)
DECODER_LAYER_RE = re.compile(
    r"\[CoreMLProbe\] Decoder layer \d+-\d+ duration=(?P<seconds>[0-9.]+)s"
)
LM_HEAD_TOKEN_RE = re.compile(
    r"\[CoreMLProbe\] LM head token (?P<index>\d+) duration=(?P<seconds>[0-9.]+)s"
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


def run_summaries(coreml_lines: list[str]) -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    current_token_index: int | None = None

    def start_run(detail: str = "") -> dict[str, object]:
        return {
            "detail": detail,
            "requested_tokens": None,
            "retain_decoders": None,
            "token_totals": [],
            "token_breakdowns": {},
            "peak_memory": None,
            "finished": None,
            "failed": None,
        }

    def token_breakdown(summary: dict[str, object], token_index: int) -> dict[str, float | None]:
        token_breakdowns = summary["token_breakdowns"]
        assert isinstance(token_breakdowns, dict)
        if token_index not in token_breakdowns:
            token_breakdowns[token_index] = {
                "total": None,
                "embedding": 0.0,
                "decoder_load": 0.0,
                "decoder_predict": 0.0,
                "lm_head": 0.0,
            }
        value = token_breakdowns[token_index]
        assert isinstance(value, dict)
        return value

    for line in coreml_lines:
        if match := RUN_STARTED_RE.search(line):
            if current is not None:
                runs.append(current)
            current = start_run(match.group("detail"))
            current_token_index = None
            continue

        if current is None and GENERATION_LOOP_RE.search(line):
            current = start_run()
            current_token_index = None

        if current is None:
            continue

        if match := PROMPT_IDS_RE.search(line):
            current_token_index = int(match.group("index"))
            token_breakdown(current, current_token_index)
        elif match := GENERATION_LOOP_RE.search(line):
            current["requested_tokens"] = int(match.group("tokens"))
            current["retain_decoders"] = int(match.group("retain"))
        elif match := EMBEDDING_TOKEN_RE.search(line):
            breakdown = token_breakdown(current, int(match.group("index")))
            breakdown["embedding"] = float(breakdown["embedding"] or 0.0) + float(match.group("seconds"))
        elif match := DECODER_LOAD_RE.search(line):
            if current_token_index is not None:
                breakdown = token_breakdown(current, current_token_index)
                breakdown["decoder_load"] = float(breakdown["decoder_load"] or 0.0) + float(match.group("seconds"))
        elif match := DECODER_LAYER_RE.search(line):
            if current_token_index is not None:
                breakdown = token_breakdown(current, current_token_index)
                breakdown["decoder_predict"] = float(breakdown["decoder_predict"] or 0.0) + float(match.group("seconds"))
        elif match := LM_HEAD_TOKEN_RE.search(line):
            breakdown = token_breakdown(current, int(match.group("index")))
            breakdown["lm_head"] = float(breakdown["lm_head"] or 0.0) + float(match.group("seconds"))
        elif match := TOKEN_TOTAL_RE.search(line):
            token_totals = current["token_totals"]
            assert isinstance(token_totals, list)
            token_index = int(match.group("index"))
            seconds = float(match.group("seconds"))
            token_totals.append(
                {
                    "index": token_index,
                    "seconds": seconds,
                    "memory": float(match.group("memory")),
                }
            )
            breakdown = token_breakdown(current, token_index)
            breakdown["total"] = seconds
        elif match := PEAK_MEMORY_RE.search(line):
            current["peak_memory"] = float(match.group("memory"))
        elif match := RUN_FINISHED_RE.search(line):
            current["finished"] = match.group("summary")
            runs.append(current)
            current = None
            current_token_index = None
        elif match := RUN_FAILED_RE.search(line):
            current["failed"] = match.group("summary")
            runs.append(current)
            current = None
            current_token_index = None

    if current is not None:
        runs.append(current)
    return runs


def format_warm_average(token_totals: list[dict[str, float]]) -> str:
    if len(token_totals) <= 1:
        return "-"
    warm_seconds = [token["seconds"] for token in token_totals[1:]]
    return f"{sum(warm_seconds) / len(warm_seconds):.2f}s"


def format_timing_breakdown(
    token_totals: list[dict[str, float]],
    token_breakdowns: dict[int, dict[str, float | None]],
    *,
    warm_only: bool,
) -> str | None:
    selected_indices = [
        int(token["index"])
        for offset, token in enumerate(token_totals)
        if not warm_only or offset > 0
    ]
    selected = [
        token_breakdowns[index]
        for index in selected_indices
        if index in token_breakdowns and token_breakdowns[index]["total"] is not None
    ]
    if not selected:
        return None

    count = len(selected)
    total = sum(float(item["total"] or 0.0) for item in selected)
    embedding = sum(float(item["embedding"] or 0.0) for item in selected)
    decoder_load = sum(float(item["decoder_load"] or 0.0) for item in selected)
    decoder_predict = sum(float(item["decoder_predict"] or 0.0) for item in selected)
    lm_head = sum(float(item["lm_head"] or 0.0) for item in selected)
    other = total - embedding - decoder_load - decoder_predict - lm_head
    label = "warm" if warm_only else "all"
    return (
        f"{label} timing avg over {count} token(s): "
        f"total={total / count:.2f}s, "
        f"decoder_load={decoder_load / count:.2f}s, "
        f"decoder_predict={decoder_predict / count:.2f}s, "
        f"lm_head={lm_head / count:.2f}s, "
        f"embedding={embedding / count:.3f}s, "
        f"other={other / count:.2f}s"
    )


def first_match(pattern: re.Pattern[str], text: str) -> re.Match[str] | None:
    return pattern.search(text)


def last_match(pattern: re.Pattern[str], text: str) -> re.Match[str] | None:
    matches = list(pattern.finditer(text))
    return matches[-1] if matches else None


def analyze(text: str, args: argparse.Namespace) -> int:
    reporter = Reporter()
    coreml_lines = [line for line in text.splitlines() if "[CoreMLProbe]" in line]

    if not coreml_lines:
        reporter.error("no [CoreMLProbe] lines found")
    else:
        reporter.ok(f"found {len(coreml_lines)} CoreMLProbe log lines")

    summaries = run_summaries(coreml_lines)
    for index, summary in enumerate(summaries, start=1):
        token_totals = summary["token_totals"]
        assert isinstance(token_totals, list)
        peak = summary["peak_memory"]
        peak_text = f"{peak:.1f} MB" if isinstance(peak, float) else "not recorded"
        status = "finished" if summary["finished"] else "failed" if summary["failed"] else "incomplete"
        reporter.ok(
            "run "
            f"{index}: status={status} retain_decoders={summary['retain_decoders']} "
            f"tokens_seen={len(token_totals)}/{summary['requested_tokens']} "
            f"warm_avg={format_warm_average(token_totals)} peak={peak_text}"
        )
        token_breakdowns = summary["token_breakdowns"]
        assert isinstance(token_breakdowns, dict)
        for warm_only in (True, False):
            timing = format_timing_breakdown(token_totals, token_breakdowns, warm_only=warm_only)
            if timing:
                reporter.ok(f"run {index}: {timing}")
        if status == "incomplete":
            reporter.warn(
                f"run {index} did not reach run finished/run failed; log may be partial or the app may have stalled"
            )

    run_started = last_match(RUN_STARTED_RE, text)
    if run_started:
        detail = run_started.group("detail")
        reporter.ok(f"run started: {detail}")
    else:
        reporter.warn("missing run started line")

    run_finished = last_match(RUN_FINISHED_RE, text)
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

    peak_match = last_match(PEAK_MEMORY_RE, text)
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

    generated_match = last_match(GENERATED_TOKENS_RE, text)
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
