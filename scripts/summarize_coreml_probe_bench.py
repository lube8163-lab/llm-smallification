#!/usr/bin/env python3
"""Summarize CoreMLProbe device-console logs into per-run speed/memory metrics.

Reads one or more `device-console.log` files (or a suite directory that
contains `<kind>/device-console.log` files produced by
run_coreml_probe_device_automation.sh) and prints a compact markdown summary:

  * warm sec/token (mean of per-token totals excluding the first, cold token)
  * cold (token 1) seconds
  * peak memory
  * for generate runs: retained-decoder count and any memory-margin stop
  * for memory-ramp runs: how many chunks stayed resident and the stop reason

The point is to surface "where the time/memory went" after a change without
hand-reading the raw console log. Pure stdlib, no external deps.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

TOKEN_TOTAL = re.compile(r"Token (\d+) total duration=([\d.]+)s")
PEAK = re.compile(r"Peak memory .*memory=([\d.]+) MB")
RAMP_RESULT = re.compile(r"Memory ramp result .*detail=(.*)$")
RETAIN_STOP = re.compile(r"Retain stop .*detail=(.*)$")
RAMP_STOP = re.compile(r"Ramp stop .*detail=(.*)$")
RETAINING = re.compile(r"Retain decoder models .*retaining=(\d+)")
RETAINED_NAMES = re.compile(r"Retained decoder models .*detail=(.*)$")
GEN_LOOP_RETAIN = re.compile(r"retain decoders=(\d+)")
SWEEP_CASE = re.compile(r"sweep case started (.*)$")
RUN_FINISHED = re.compile(r"run finished (.*)$")
AUTO_FINISH = re.compile(r"automation finished .*summary=(.*)$")
RUN_FAILED = re.compile(r"run failed: (.*)$")
ERR_CODE = re.compile(r"error code: (-?\d+)")
MIL_BNNS = re.compile(r"compiling MIL to BNNS graph")


@dataclass
class Segment:
    label: str
    token_secs: dict[int, float] = field(default_factory=dict)
    peak_mb: float | None = None
    retain_requested: int | None = None
    retain_actual: int | None = None
    retain_stop: str | None = None
    ramp_result: str | None = None
    ramp_stop: str | None = None
    finished: str | None = None
    error: str | None = None

    @property
    def warm_avg(self) -> float | None:
        warm = [s for i, s in self.token_secs.items() if i > 1]
        if warm:
            return sum(warm) / len(warm)
        if self.token_secs:
            return next(iter(self.token_secs.values()))
        return None

    @property
    def cold(self) -> float | None:
        return self.token_secs.get(1)


def parse_log(text: str) -> list[Segment]:
    segments: list[Segment] = []
    current = Segment(label="run")
    segments.append(current)

    for line in text.splitlines():
        m = SWEEP_CASE.search(line)
        if m:
            detail = m.group(1)
            retain = None
            rm = re.search(r"retain=(\d+)", detail)
            if rm:
                retain = int(rm.group(1))
            label = detail
            dm = re.search(r"decoder=(\S+)", detail)
            if dm and retain is not None:
                label = f"decoder={dm.group(1)} retain={retain}"
            current = Segment(label=label, retain_requested=retain)
            segments.append(current)
            continue

        m = TOKEN_TOTAL.search(line)
        if m:
            current.token_secs[int(m.group(1))] = float(m.group(2))
            continue
        m = PEAK.search(line)
        if m:
            current.peak_mb = float(m.group(1))
            continue
        m = RETAINING.search(line)
        if m:
            current.retain_requested = current.retain_requested or int(m.group(1))
            continue
        m = GEN_LOOP_RETAIN.search(line)
        if m and current.retain_requested is None:
            current.retain_requested = int(m.group(1))
            continue
        m = RETAINED_NAMES.search(line)
        if m:
            names = m.group(1)
            if "none" in names:
                current.retain_actual = 0
            else:
                current.retain_actual = len([n for n in names.split(",") if n.strip()])
            continue
        m = RETAIN_STOP.search(line)
        if m:
            current.retain_stop = m.group(1)
            continue
        m = RAMP_RESULT.search(line)
        if m:
            current.ramp_result = m.group(1)
            continue
        m = RAMP_STOP.search(line)
        if m:
            current.ramp_stop = m.group(1)
            continue
        m = RUN_FAILED.search(line)
        if m:
            code = ERR_CODE.search(m.group(1))
            if MIL_BNNS.search(text) and code:
                current.error = f"execution-plan compile fail ({code.group(1)}, BNNS)"
            elif code:
                current.error = f"fail (code {code.group(1)})"
            else:
                current.error = "fail"
            continue
        m = RUN_FINISHED.search(line) or AUTO_FINISH.search(line)
        if m:
            current.finished = m.group(1)

    # Drop the leading empty pre-sweep segment when sweep cases exist.
    non_empty = [s for s in segments if s.token_secs or s.ramp_result or s.retain_stop or s.peak_mb]
    return non_empty or segments


def fmt(value: float | None, suffix: str = "") -> str:
    if value is None:
        return "-"
    return f"{value:.2f}{suffix}"


def summarize_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    kind = path.parent.name
    rows: list[str] = []
    for seg in parse_log(text):
        if seg.ramp_result is not None:
            rows.append(
                f"| {kind} | ramp | - | - | {fmt(seg.peak_mb,' MB')} | {seg.ramp_result} |"
            )
            continue
        detail_bits = []
        if seg.retain_requested is not None:
            actual = seg.retain_actual
            if actual is not None and actual != seg.retain_requested:
                detail_bits.append(f"retain {actual}/{seg.retain_requested} (valve)")
            else:
                detail_bits.append(f"retain {seg.retain_requested}")
        if seg.retain_stop:
            detail_bits.append("MARGIN STOP")
        if seg.error:
            detail_bits.append(f"❌ {seg.error}")
        detail = "; ".join(detail_bits) if detail_bits else (seg.finished or "-")
        rows.append(
            f"| {kind} | {seg.label} | {fmt(seg.warm_avg,'s')} | {fmt(seg.cold,'s')} | "
            f"{fmt(seg.peak_mb,' MB')} | {detail} |"
        )
    if not rows:
        rows.append(f"| {kind} | (no metrics parsed) | - | - | - | - |")
    return rows


def collect_logs(args: list[str]) -> list[Path]:
    logs: list[Path] = []
    for arg in args:
        p = Path(arg)
        if p.is_dir():
            found = sorted(p.glob("*/device-console.log"))
            if not found:
                found = sorted(p.glob("**/device-console.log"))
            logs.extend(found)
        elif p.is_file():
            logs.append(p)
        else:
            print(f"warning: not found: {arg}", file=sys.stderr)
    return logs


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        print("usage: summarize_coreml_probe_bench.py <suite-dir | console.log> ...")
        return 2

    logs = collect_logs(argv)
    if not logs:
        print("no console logs found", file=sys.stderr)
        return 1

    print("## CoreMLProbe bench summary")
    print()
    print("| kind | run | warm/token | cold(t1) | peak | detail |")
    print("|---|---|---:|---:|---:|---|")
    for log in logs:
        for row in summarize_file(log):
            print(row)
    print()
    print(f"_source: {len(logs)} console log(s)_")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
