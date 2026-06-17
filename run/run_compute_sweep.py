#!/usr/bin/env python3
"""Sweep --compute-per-{k,v}-line x core count on the b=1 KV decode trace.

compute-per-line sets the instruction distance between consecutive memory
instructions (compute + 1 insts/line), which against SimpleO3's 128-deep
instruction window determines memory-level parallelism and therefore each
core's bandwidth demand: demand ~ 1/(compute+1) while window-limited
(measured at n=1: 5.66 / 3.16 / 1.62 / 0.82 GB/s for 16/32/64/128; the c16
point is already sub-proportional -- memory-side limits begin to bind).
Multi-core points then show where each demand level saturates the single
DDR4-2400 channel.

Per-core address layout replicates run_kv_core_batch_sweep.py (disjoint
33 MiB regions: K_i = base + i*region, all V above all K), so rows at the
default compute (32) reproduce core_sweep_b1 points exactly. Non-pow2 core
counts run with llc_associativity = n_cores (SimpleO3 pow2-set workaround;
see the sweep script docstring).

Results merge into <out-dir>/compute_sweep_summary.csv keyed by
(compute, cores); existing keys are skipped, so re-runs only fill gaps.
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ram21  # noqa: E402
from run_kv_core_batch_sweep import (  # noqa: E402
    CPU_HZ, INSTS_PATTERN, LLC_DEFAULT_ASSOC, llc_auto_assoc, parse_stats,
    summarize, DEFAULT_BASE_K, REGION_ALIGN,
)

COLS = ["compute", "cores", "insts", "runtime_cycles", "per_core_ipc",
        "avg_read_latency_ns", "avg_read_queue_len", "row_hit_rate",
        "total_read_reqs", "read_bw_gbs", "power", "llc_assoc"]

# Llama-3.1-8B fixed point: L32, qh32/kvh8, d128, ctx512, t1, b1, fp16.
SHAPE = dict(num_layers=32, num_query_heads=32, num_kv_heads=8, head_dim=128,
             context_len=512, decode_tokens=1, batch=1)
TENSOR_BYTES = (SHAPE["num_layers"] * SHAPE["batch"] * SHAPE["num_kv_heads"]
                * (SHAPE["context_len"] + SHAPE["decode_tokens"])
                * SHAPE["head_dim"] * 2)
REGION = (TENSOR_BYTES + REGION_ALIGN - 1) // REGION_ALIGN * REGION_ALIGN


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--computes", type=int, nargs="+", default=[16, 32, 64, 128])
    p.add_argument("--cores", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--out-dir", type=Path, default=Path("results/compute_sweep"))
    p.add_argument("--trace-dir", type=Path, default=Path("traces/compute_sweep"))
    p.add_argument("--gen-script", type=Path, default=Path("gen/gen_kv_decode_trace.py"))
    p.add_argument("--inst-slack", type=float, default=1.1)
    p.add_argument("--gqa-emission", choices=["reread", "naive"], default=None,
                   help="Forwarded to the generator (default reread); "
                        "'naive' for 1:1 emission-mode comparison.")
    return p.parse_args()


def gen_trace(args, compute: int, path: Path, base_k: int, base_v: int) -> int:
    cmd = [sys.executable, str(args.gen_script),
           "--num-layers", str(SHAPE["num_layers"]),
           "--num-query-heads", str(SHAPE["num_query_heads"]),
           "--num-kv-heads", str(SHAPE["num_kv_heads"]),
           "--head-dim", str(SHAPE["head_dim"]),
           "--context-len", str(SHAPE["context_len"]),
           "--decode-tokens", str(SHAPE["decode_tokens"]),
           "--batch", str(SHAPE["batch"]),
           "--compute-per-k-line", str(compute),
           "--compute-per-v-line", str(compute),
           "--base-k", hex(base_k), "--base-v", hex(base_v),
           "--out", str(path)]
    if args.gqa_emission is not None:
        cmd += ["--gqa-emission", args.gqa_emission]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return int(INSTS_PATTERN.search(proc.stdout).group(1))


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.trace_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "compute_sweep_summary.csv"

    rows: list[dict] = []
    if csv_path.exists():
        rows = list(csv.DictReader(csv_path.open()))
    done = {(int(r["compute"]), int(r["cores"])) for r in rows}

    def write_csv() -> None:
        rows.sort(key=lambda r: (int(r["compute"]), int(r["cores"])))
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in COLS})

    for compute in args.computes:
        for n in args.cores:
            if (compute, n) in done:
                print(f"skip compute={compute} cores={n} (already in CSV)", flush=True)
                continue
            print(f"\n========== compute={compute} cores={n} ==========", flush=True)
            base_v0 = DEFAULT_BASE_K + n * REGION
            traces, insts = [], None
            for i in range(n):
                tp = args.trace_dir / (
                    f"kv_b1_n{n}_core{i}_c{SHAPE['context_len']}"
                    f"_t{SHAPE['decode_tokens']}_L{SHAPE['num_layers']}_comp{compute}.trace")
                got = gen_trace(args, compute, tp,
                                DEFAULT_BASE_K + i * REGION, base_v0 + i * REGION)
                if insts is None:
                    insts = got
                elif insts != got:
                    raise RuntimeError(f"inst mismatch across cores: {insts} vs {got}")
                traces.append(tp)
            # methodology v2: minimal pow2-set ways >= 6 (pow2 n -> 8 = default).
            llc_assoc = llc_auto_assoc(n)
            num_expected = int(insts * args.inst_slack)
            print(f"  insts={insts:,} llc_assoc={llc_assoc or 'default'}", flush=True)
            sim = ram21.build_sim("ddr4", traces, num_expected_insts=num_expected,
                                  llc_associativity=llc_assoc)
            sp = args.out_dir / f"ddr4_b1_n{n}_comp{compute}.stats"
            ram21.run_to_stats_file(sim, sp)
            s = summarize(parse_stats(sp), expected_cores=n, per_core_insts=insts)
            tu = ram21.MEMORIES["ddr4"]["time_unit_ns"]
            lat_ns = (s["avg_read_latency"] * tu
                      if not math.isnan(s["avg_read_latency"]) else math.nan)
            bw = (s["total_read_reqs"] * ram21.CACHE_LINE
                  / (s["runtime_cycles"] / CPU_HZ) / 1e9)
            row = {"compute": compute, "cores": n, "insts": insts,
                   "runtime_cycles": s["runtime_cycles"],
                   "per_core_ipc": s["per_core_ipc"],
                   "avg_read_latency_ns": lat_ns,
                   "avg_read_queue_len": s["avg_read_queue_len"],
                   "row_hit_rate": s["row_hit_rate"],
                   "total_read_reqs": s["total_read_reqs"],
                   "read_bw_gbs": bw,
                   "power": bw / lat_ns if lat_ns and not math.isnan(lat_ns) else math.nan,
                   "llc_assoc": llc_assoc if llc_assoc is not None else LLC_DEFAULT_ASSOC}
            rows.append(row)
            done.add((compute, n))
            write_csv()
            print(f"  runtime={s['runtime_cycles']:,} IPC={s['per_core_ipc']:.3f} "
                  f"lat={lat_ns:.1f}ns qlen={s['avg_read_queue_len']:.2f} "
                  f"row_hit={s['row_hit_rate']:.4f} total_BW={bw:.2f} GB/s", flush=True)

    print(f"\nSummary CSV: {csv_path}")


if __name__ == "__main__":
    main()
