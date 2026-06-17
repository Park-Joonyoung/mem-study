#!/usr/bin/env python3
"""Compare Ramulator2 stats files.

Ramulator2's output is YAML-like, but repeated Controller sections make it
awkward to parse as plain YAML. This script uses a small line-based parser and
then derives the metrics we care about for DDR workload comparisons.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from statistics import mean, pstdev


# Real time per memory simulation tick, per standard (tCK_ps / 1000). Used to
# convert memory-cycle latency to ns.
TIME_UNIT_NS = {"DDR4": 0.833, "DDR5": 0.833}


CHANNEL_METRICS = {
    "avg_read_latency",
    "read_queue_len_avg",
    "write_queue_len",
    "queue_len",
    "num_other_reqs",
    "num_write_reqs",
    "read_latency",
    "priority_queue_len_avg",
    "row_hits",
    "priority_queue_len",
    "row_misses",
    "row_conflicts",
    "read_row_misses",
    "queue_len_avg",
    "read_row_conflicts_core",
    "read_row_hits",
    "write_queue_len_avg",
    "read_row_conflicts",
    "write_row_misses",
    "write_row_conflicts",
    "read_queue_len",
    "write_row_hits",
    "read_row_hits_core",
    "read_row_misses_core",
    "num_read_reqs",
}

SUM_CHANNEL_METRICS = [
    "num_read_reqs",
    "num_write_reqs",
    "read_latency",
    "row_hits",
    "row_misses",
    "row_conflicts",
    "read_row_hits",
    "read_row_misses",
    "read_row_conflicts",
    "write_row_hits",
    "write_row_misses",
    "write_row_conflicts",
    "queue_len",
    "read_queue_len",
    "write_queue_len",
    "priority_queue_len",
]


def parse_value(raw: str):
    token = raw.strip().split()[0] if raw.strip() else ""
    if not token:
        raise ValueError("empty value")
    if token == ".nan":
        return math.nan
    if any(c in token for c in ".eE"):
        return float(token)
    return int(token)


def parse_stats(path: Path):
    globals_: dict[str, float | int] = {}
    channels: dict[int, dict[str, float | int]] = {}

    for line in path.read_text().splitlines():
        if ":" not in line:
            continue

        key, raw = line.split(":", 1)
        key = key.strip()
        try:
            value = parse_value(raw)
        except ValueError:
            continue

        channel_match = re.match(r"(.+)_([0-9]+)$", key)
        if channel_match and channel_match.group(1) in CHANNEL_METRICS:
            metric = channel_match.group(1)
            channel = int(channel_match.group(2))
            channels.setdefault(channel, {})[metric] = value
        else:
            globals_[key] = value

    return globals_, channels


def safe_div(num: float, den: float) -> float:
    if den == 0 or math.isnan(den):
        return math.nan
    return num / den


def summarize(path: Path):
    globals_, channels = parse_stats(path)
    summary: dict[str, float | int] = dict(globals_)

    for metric in SUM_CHANNEL_METRICS:
        summary[f"{metric}_sum"] = sum(channel.get(metric, 0) for channel in channels.values())

    expected = float(summary.get("num_expected_insts", math.nan))
    cycles = float(summary.get("cycles_recorded_core_0", math.nan))
    summary["ipc"] = safe_div(expected, cycles)

    read_reqs = float(summary.get("num_read_reqs_sum", 0))
    read_latency = float(summary.get("read_latency_sum", 0))
    summary["avg_read_latency_weighted"] = safe_div(read_latency, read_reqs)

    row_total = (
        float(summary.get("row_hits_sum", 0))
        + float(summary.get("row_misses_sum", 0))
        + float(summary.get("row_conflicts_sum", 0))
    )
    summary["row_hit_rate"] = safe_div(float(summary.get("row_hits_sum", 0)), row_total)

    read_row_total = (
        float(summary.get("read_row_hits_sum", 0))
        + float(summary.get("read_row_misses_sum", 0))
        + float(summary.get("read_row_conflicts_sum", 0))
    )
    summary["read_row_hit_rate"] = safe_div(float(summary.get("read_row_hits_sum", 0)), read_row_total)

    llc_read_access = float(summary.get("llc_read_access", 0))
    llc_write_access = float(summary.get("llc_write_access", 0))
    summary["llc_read_miss_rate"] = safe_div(float(summary.get("llc_read_misses", 0)), llc_read_access)
    summary["llc_write_miss_rate"] = safe_div(float(summary.get("llc_write_misses", 0)), llc_write_access)

    total_reqs = float(summary.get("total_num_read_requests", 0)) + float(summary.get("total_num_write_requests", 0))
    summary["dram_traffic_mib"] = total_reqs * 64 / (1024 * 1024)

    active_channels = [
        channel
        for channel, stats in sorted(channels.items())
        if stats.get("num_read_reqs", 0) or stats.get("num_write_reqs", 0)
    ]
    summary["active_channels"] = len(active_channels)

    active_reads = [float(channels[channel].get("num_read_reqs", 0)) for channel in active_channels]
    if active_reads:
        avg_reads = mean(active_reads)
        summary["channel_read_max_over_avg"] = safe_div(max(active_reads), avg_reads)
        summary["channel_read_cv"] = safe_div(pstdev(active_reads), avg_reads)
    else:
        summary["channel_read_max_over_avg"] = math.nan
        summary["channel_read_cv"] = math.nan

    for avg_metric in ["queue_len_avg", "read_queue_len_avg", "write_queue_len_avg", "priority_queue_len_avg"]:
        weighted_sum = 0.0
        weight_sum = 0.0
        for stats in channels.values():
            weight = float(stats.get("num_read_reqs", 0)) + float(stats.get("num_write_reqs", 0))
            if weight:
                weighted_sum += float(stats.get(avg_metric, 0)) * weight
                weight_sum += weight
        summary[f"{avg_metric}_req_weighted"] = safe_div(weighted_sum, weight_sum)

    return summary, channels


def infer_label(path: Path) -> str:
    name = path.stem.lower()
    if "ddr4" in name:
        return "DDR4"
    return path.stem


def format_value(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if abs(value) >= 1_000:
            return f"{value:,.0f}"
        if abs(value) >= 10:
            return f"{value:,.2f}"
        if abs(value) >= 1:
            return f"{value:,.3f}"
        return f"{value:.4f}"
    return str(value)


def print_table(labels: list[str], summaries: list[dict[str, float | int]]) -> None:
    metrics = [
        ("Instructions", "num_expected_insts"),
        ("CPU cycles", "cycles_recorded_core_0"),
        ("IPC", "ipc"),
        ("Memory access cycles", "memory_access_cycles_recorded_core_0"),
        ("LLC read miss rate", "llc_read_miss_rate"),
        ("LLC write miss rate", "llc_write_miss_rate"),
        ("DRAM read requests", "total_num_read_requests"),
        ("DRAM write requests", "total_num_write_requests"),
        ("DRAM traffic MiB", "dram_traffic_mib"),
        ("Avg read latency (ns)", "avg_read_latency_ns"),
        ("Row hit rate", "row_hit_rate"),
        ("Read row hit rate", "read_row_hit_rate"),
        ("Row conflicts", "row_conflicts_sum"),
        ("Req-weighted queue avg", "queue_len_avg_req_weighted"),
        ("Active channels", "active_channels"),
        ("Channel read max/avg", "channel_read_max_over_avg"),
        ("Channel read CV", "channel_read_cv"),
    ]

    include_ratio = len(summaries) == 2
    header = ["metric", *labels]
    if include_ratio:
        header.extend([f"{labels[1]}/{labels[0]}", f"{labels[0]}/{labels[1]}"])

    rows: list[list[str]] = []
    for metric_name, key in metrics:
        values = [summary.get(key) for summary in summaries]
        row = [metric_name, *[format_value(value) for value in values]]
        if include_ratio:
            left = float(values[0]) if values[0] is not None else math.nan
            right = float(values[1]) if values[1] is not None else math.nan
            row.append(format_value(safe_div(right, left)))
            row.append(format_value(safe_div(left, right)))
        rows.append(row)

    widths = [len(item) for item in header]
    for row in rows:
        widths = [max(width, len(item)) for width, item in zip(widths, row)]

    print("  ".join(item.ljust(widths[i]) if i == 0 else item.rjust(widths[i]) for i, item in enumerate(header)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(item.ljust(widths[i]) if i == 0 else item.rjust(widths[i]) for i, item in enumerate(row)))


def print_channels(labels: list[str], channels_list: list[dict[int, dict[str, float | int]]]) -> None:
    for label, channels in zip(labels, channels_list):
        active = [
            (channel, stats)
            for channel, stats in sorted(channels.items())
            if stats.get("num_read_reqs", 0) or stats.get("num_write_reqs", 0)
        ]
        if not active:
            continue

        print(f"\n{label} active channels")
        print(f"{'ch':>3}  {'reads':>12}  {'writes':>8}  {'avg_lat':>9}  {'row_hit%':>9}")
        print(f"{'-' * 3}  {'-' * 12}  {'-' * 8}  {'-' * 9}  {'-' * 9}")
        for channel, stats in active:
            reads = float(stats.get("num_read_reqs", 0))
            writes = float(stats.get("num_write_reqs", 0))
            avg_lat = float(stats.get("avg_read_latency", math.nan))
            row_total = (
                float(stats.get("row_hits", 0))
                + float(stats.get("row_misses", 0))
                + float(stats.get("row_conflicts", 0))
            )
            row_hit_pct = safe_div(float(stats.get("row_hits", 0)), row_total) * 100
            print(f"{channel:>3}  {reads:>12,.0f}  {writes:>8,.0f}  {avg_lat:>9.2f}  {row_hit_pct:>8.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stats", nargs="+", type=Path, help="Ramulator2 stats files to compare")
    parser.add_argument("--labels", nargs="+", help="Optional labels, one per stats file")
    parser.add_argument("--no-channels", action="store_true", help="Hide per-channel read/write breakdown")
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.stats):
        parser.error("--labels must have the same length as stats files")

    labels = args.labels or [infer_label(path) for path in args.stats]
    summaries = []
    channels_list = []
    for path, label in zip(args.stats, labels):
        summary, channels = summarize(path)
        # Domain-safe latency: convert memory cycles -> ns using the standard's
        # tick period. memory_system_cycles (memory-clock domain) is dropped from
        # the table; use CPU cycles / IPC for cross-standard speed comparisons.
        tu = TIME_UNIT_NS.get(label)
        lat = float(summary.get("avg_read_latency_weighted", math.nan))
        summary["avg_read_latency_ns"] = lat * tu if tu is not None else math.nan
        summaries.append(summary)
        channels_list.append(channels)

    print_table(labels, summaries)
    if not args.no_channels:
        print_channels(labels, channels_list)


if __name__ == "__main__":
    main()
