#!/usr/bin/env python3
"""Core sweep over EXPANDED REAL llama.cpp KV traces (see gen/expand_kv_log.py).

Counterpart of run_kv_core_batch_sweep.py for real-capture traces: each core
replays the same expanded decode stream in its own disjoint address region
(exactly like the synthetic sweep, where per-core traces differ only by base
address), so core count remains the pure concurrency/contention axis and rows
are directly comparable with the synthetic core_sweep_b1 results.

Input is the expander's meta.json (trace file list + per-core inst count).
Output CSV schema matches core_batch_sweep_summary.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ram21  # noqa: E402
from run_kv_core_batch_sweep import (  # noqa: E402
    CPU_HZ, LLC_DEFAULT_ASSOC, llc_auto_assoc, parse_stats, summarize,
)

COLS = ["batch", "cores", "dram",
        "runtime_cycles", "per_core_ipc",
        "n_cores_observed", "n_cores_expected",
        "active_channels", "total_channels",
        "avg_read_latency_ns", "avg_read_queue_len", "row_hit_rate",
        "total_read_reqs", "read_bw_gbs", "power", "llc_assoc"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--meta", type=Path, required=True,
                   help="meta.json written by gen/expand_kv_log.py")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--cores", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--dram", nargs="+", choices=sorted(ram21.MEMORIES),
                   default=["ddr4"])
    p.add_argument("--inst-slack", type=float, default=1.1)
    p.add_argument("--llc-associativity", default=None,
                   help="LLC ways: an int, or 'auto' (= n_cores); see "
                        "run_kv_core_batch_sweep.py.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    meta = json.loads(args.meta.read_text())
    files = [Path(f) for f in meta["files"]]
    per_core_insts = int(meta["per_core_insts"])
    tag = meta["tag"]
    for f in files:
        if not f.exists():
            sys.stderr.write(f"ERROR: trace missing: {f}\n"); sys.exit(1)
    if max(args.cores) > len(files):
        sys.stderr.write(f"ERROR: need {max(args.cores)} per-core traces, "
                         f"meta has {len(files)} (re-run expander with "
                         f"--cores {max(args.cores)}).\n"); sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "real_core_sweep_summary.csv"
    rows: list[dict] = []

    def write_csv() -> None:
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in COLS})

    num_expected = int(per_core_insts * args.inst_slack)
    print(f"traces: {tag}, per_core_insts={per_core_insts:,}, "
          f"num_expected_insts={num_expected:,}")

    for n_cores in args.cores:
        for dram_name in args.dram:
            llc_assoc = None
            if args.llc_associativity is not None:
                llc_assoc = (llc_auto_assoc(n_cores)
                             if args.llc_associativity == "auto"
                             else int(args.llc_associativity))
            stats_path = args.out_dir / f"{dram_name}_{tag}_b1_n{n_cores}.stats"
            print(f"  running {dram_name} n={n_cores} "
                  f"{'' if llc_assoc is None else f'(llc_assoc={llc_assoc}) '}...",
                  flush=True)
            sim = ram21.build_sim(
                dram_name, files[:n_cores], num_expected_insts=num_expected,
                llc_associativity=llc_assoc,
            )
            ram21.run_to_stats_file(sim, stats_path)
            metrics = parse_stats(stats_path)
            s = summarize(metrics, expected_cores=n_cores,
                          per_core_insts=per_core_insts)
            s["batch"] = 1
            s["cores"] = n_cores
            s["dram"] = dram_name
            tu = ram21.MEMORIES[dram_name]["time_unit_ns"]
            s["avg_read_latency_ns"] = (
                s["avg_read_latency"] * tu
                if not math.isnan(s["avg_read_latency"]) else math.nan)
            s["read_bw_gbs"] = (
                s["total_read_reqs"] * ram21.CACHE_LINE
                / (s["runtime_cycles"] / CPU_HZ) / 1e9
                if s["runtime_cycles"] and s["total_read_reqs"] is not None
                else math.nan)
            s["power"] = (
                s["read_bw_gbs"] / s["avg_read_latency_ns"]
                if not (math.isnan(s["read_bw_gbs"]) or math.isnan(s["avg_read_latency_ns"]))
                and s["avg_read_latency_ns"] else math.nan)
            s["llc_assoc"] = llc_assoc if llc_assoc is not None else LLC_DEFAULT_ASSOC
            rows.append(s)
            write_csv()
            print(
                f"    runtime={s['runtime_cycles']:>14,}  "
                f"IPC={s['per_core_ipc']:.3f}  "
                f"cores={s['n_cores_observed']}/{s['n_cores_expected']}  "
                f"lat={s['avg_read_latency_ns']:.1f}ns  "
                f"rd_qlen={s['avg_read_queue_len']:.2f}  "
                f"row_hit={s['row_hit_rate']:.4f}"
            )

    print(f"\nSummary CSV: {csv_path}")


if __name__ == "__main__":
    main()
