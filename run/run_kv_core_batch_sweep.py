#!/usr/bin/env python3
"""2-D core x batch sweep on Llama-3.1-8B-style KV decode.

Two independent axes:
  --cores N   number of CONCURRENT cores (independent decode streams). The
              concurrency / contention axis: SimpleO3 runs one core per trace,
              all sharing the memory subsystem.
  --batch B   per-trace LLM batch dimension, forwarded to the generator. Within
              one core a batch of B sequences is walked SERIALLY in a single
              instruction stream (layout [layer][batch][head][seq][dim]); it
              raises per-core footprint / working set and changes row-buffer
              locality, but does NOT add concurrency.

For each (B, N) grid point, N independent traces are generated with DISTINCT,
auto-sized base addresses (one disjoint K and V region per core, sized to hold
that batch's tensor so cores never alias) and run as N SimpleO3 cores.
Contention comes from N; B changes per-stream intensity. Holding N*B fixed
trades "many light cores" vs "few heavy cores".

REPORTED METRICS:
  memory_system_cycles    Ramulator2 sim length, primary speedup metric
  runtime_cycles          max(per-core cycles), wall-clock at the slowest core
  per_core_ipc            per_core_insts / max(per-core cycles); throughput
                          per core. Drops as memory saturates.
  active_channels         channels with non-zero read activity
  avg_read_latency        read_latency / num_read_reqs (per channel, then
                          averaged over active channels)
  avg_read_queue_len      time-averaged per-channel read queue length; a value
                          pinned at the queue capacity is a saturation signal
  row_hit_rate            read_row_hits / (hits + misses + conflicts)

NOTE (Ramulator 2.0 -> 2.1): older 2.0 runs had an inflated per-channel
`num_read_reqs` counter under saturated multi-core, so this script used to
recompute counts/latency from the row counters. Ramulator 2.1 (driven via
ram21.py) reports `num_read_reqs` consistent with `total_num_read_requests`
and the row-counter sum, so we now read the native counters directly. A cheap
divergence sentinel in summarize() warns if that assumption ever breaks again.

ADDRESS-SPACE LAYOUT (auto-sized per grid point):
  region    = align_up(per-core K-tensor bytes, 1 MiB)   (>= one core's KV)
  K_base[i] = --base-k + i * region
  V_base[i] = (--base-k + N * region) + i * region       (all V above all K)
  Each core's K and V tensor fits its own disjoint region, so no two cores
  alias. --core-region-size, if given, overrides `region` (validated >= tensor).
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ram21  # noqa: E402  (Ramulator 2.1 Python-API runner; puts repo root on sys.path)
# Single-source simulation constants (coupling notes in sim_constants.py);
# re-exported from this module for the other run/ scripts.
from sim_constants import (  # noqa: E402,F401
    CPU_HZ, DEFAULT_BASE_K, DTYPE_BYTES, LLC_DEFAULT_ASSOC, REGION_ALIGN,
    llc_auto_assoc,
)


# SimpleO3's frontend used to parse num_expected_insts as a C `int`, capping
# it at INT32_MAX (=> max batch 14 at context-len 512). Patched locally on
# 2026-06-11 (simpleO3.cpp: int -> size_t, rebuilt; validated bit-exact on the
# n8/b1 baseline). This is now only a sanity guard against runaway sim times
# (~1e11 insts ~ half a day of simulation). The binding limit on the batch
# axis is now DRAM capacity: 16 GiB fits batch<=31 at 8 cores (layout check
# in layout_for()).
MAX_EXPECTED_INSTS = 100_000_000_000

INSTS_PATTERN = re.compile(r"estimated retired instructions for one pass=(\d+)")


# =============================================================================
# Trace generation
# =============================================================================

def _align_up(n: int, a: int) -> int:
    return ((n + a - 1) // a) * a


def per_core_tensor_bytes(args, batch: int) -> int:
    """Bytes one core's K (== V) tensor spans: [layer][batch][kv_head][seq][dim]."""
    max_seq = args.context_len + args.decode_tokens
    return args.num_layers * batch * args.num_kv_heads * max_seq * args.head_dim * DTYPE_BYTES


def layout_for(args, n_cores: int, batch: int) -> tuple[int, int, int]:
    """(base_k, base_v, region) sized so each core's K/V tensor fits a disjoint
    region and all V regions sit above all K regions -> cores never alias."""
    tb = per_core_tensor_bytes(args, batch)
    if args.core_region_size is not None:
        region = args.core_region_size
        if region < tb:
            raise RuntimeError(
                f"--core-region-size {region/2**20:.1f} MiB < per-core tensor "
                f"{tb/2**20:.1f} MiB (batch={batch}); cores would alias. Raise it "
                f"or omit it to auto-size.")
    else:
        region = _align_up(tb, REGION_ALIGN)
    base_k = args.base_k
    base_v = base_k + n_cores * region   # V regions start above all K regions
    top = base_v + n_cores * region
    if top > ram21.MAX_ADDR:
        # NoTranslation wraps addresses modulo MAX_ADDR with no error, which
        # would silently alias the per-core regions.
        raise RuntimeError(
            f"layout top 0x{top:x} ({top/2**30:.2f} GiB) exceeds MAX_ADDR "
            f"{ram21.MAX_ADDR/2**30:.0f} GiB; addresses would wrap and cores "
            f"would alias. Reduce --batch/--cores/--context-len.")
    return base_k, base_v, region


def core_bases(core_id: int, base_k: int, base_v: int, region: int) -> tuple[int, int]:
    return base_k + core_id * region, base_v + core_id * region


def generate_one_trace(
    gen_script: Path, trace_path: Path,
    num_layers: int, num_query_heads: int, num_kv_heads: int,
    head_dim: int, context_len: int, decode_tokens: int, batch: int,
    base_k: int, base_v: int, gqa_emission: str | None = None,
) -> int:
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(gen_script),
        "--num-layers", str(num_layers),
        "--num-query-heads", str(num_query_heads),
        "--num-kv-heads", str(num_kv_heads),
        "--head-dim", str(head_dim),
        "--batch", str(batch),
        "--context-len", str(context_len),
        "--decode-tokens", str(decode_tokens),
        "--base-k", hex(base_k),
        "--base-v", hex(base_v),
        "--out", str(trace_path),
    ]
    if gqa_emission is not None:
        cmd += ["--gqa-emission", gqa_emission]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout); sys.stderr.write(proc.stderr)
        raise RuntimeError(f"trace generator failed (rc={proc.returncode})")
    m = INSTS_PATTERN.search(proc.stdout)
    if not m:
        sys.stderr.write(proc.stdout)
        raise RuntimeError("could not parse 'estimated retired instructions' from generator")
    return int(m.group(1))


def generate_traces(n_cores: int, batch: int, args) -> tuple[list[Path], int, int]:
    base_k, base_v, region = layout_for(args, n_cores, batch)
    paths = []
    per_core_insts = None
    for core_id in range(n_cores):
        bk, bv = core_bases(core_id, base_k, base_v, region)
        if bk + region > base_v:  # disjoint-region invariant (auto-size guarantees it)
            raise RuntimeError(
                f"core {core_id}: K region [0x{bk:x}, 0x{bk + region:x}) overruns "
                f"V_base 0x{base_v:x}.")
        trace_path = args.trace_dir / (
            f"kv_b{batch}_n{n_cores}_core{core_id}_c{args.context_len}"
            f"_t{args.decode_tokens}_L{args.num_layers}.trace"
        )
        insts = generate_one_trace(
            args.gen_script, trace_path,
            args.num_layers, args.num_query_heads, args.num_kv_heads,
            args.head_dim, args.context_len, args.decode_tokens, batch,
            bk, bv, args.gqa_emission,
        )
        paths.append(trace_path)
        if per_core_insts is None:
            per_core_insts = insts
            num_expected = int(per_core_insts * args.inst_slack)
            if num_expected > MAX_EXPECTED_INSTS:
                # Check on the first core, before generating the remaining
                # (potentially multi-GB) traces.
                raise RuntimeError(
                    f"per-core insts {per_core_insts} * slack {args.inst_slack} "
                    f"= {num_expected} exceeds SimpleO3 cap {MAX_EXPECTED_INSTS} "
                    f"(int32 param). Reduce --context-len/--batch.")
        elif per_core_insts != insts:
            raise RuntimeError(f"per-core insts mismatch: {per_core_insts} vs {insts} at core {core_id}")
    return paths, per_core_insts, region


# =============================================================================
# Stats parsing
# =============================================================================
# =============================================================================

# Per-channel integer counters.
# `num_read_reqs_<i>` is the completed-read count; `read_latency_<i>` is the sum
# of those reads' latencies (cycles), so avg = read_latency / num_read_reqs.
# `read_row_{hits,misses,conflicts}_<i>` feed row_hit_rate. (Ramulator 2.1
# reports num_read_reqs consistent with the row-counter sum; see module docstring.)
PER_CHANNEL_INT_KEYS = [
    "num_read_reqs", "num_write_reqs",
    "read_row_hits", "read_row_misses", "read_row_conflicts",
    "read_latency",
]
PER_CHANNEL_FLOAT_KEYS = [
    "read_queue_len_avg",
    "write_queue_len_avg",
    "queue_len_avg",
]
SCALAR_KEYS = [
    "memory_system_cycles",
    "total_num_read_requests",
    "total_num_write_requests",
]


def _ensure_len(lst, n, fill):
    while len(lst) <= n:
        lst.append(fill)


def parse_stats(stats_path: Path) -> dict:
    text = stats_path.read_text()
    m: dict = {f"{k}_per_channel": [] for k in PER_CHANNEL_INT_KEYS + PER_CHANNEL_FLOAT_KEYS}
    m.update({k: None for k in SCALAR_KEYS})

    for key in PER_CHANNEL_INT_KEYS:
        for mt in re.finditer(rf"^\s*{key}_(\d+):\s*(\d+)\s*$", text, re.MULTILINE):
            ch, v = int(mt.group(1)), int(mt.group(2))
            _ensure_len(m[f"{key}_per_channel"], ch, 0)
            m[f"{key}_per_channel"][ch] = v

    for key in PER_CHANNEL_FLOAT_KEYS:
        for mt in re.finditer(
            rf"^\s*{key}_(\d+):\s*(\.nan|[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*$",
            text, re.MULTILINE,
        ):
            ch = int(mt.group(1))
            raw = mt.group(2)
            v = math.nan if raw == ".nan" else float(raw)
            _ensure_len(m[f"{key}_per_channel"], ch, math.nan)
            m[f"{key}_per_channel"][ch] = v

    for key in SCALAR_KEYS:
        mt = re.search(rf"^\s*{key}:\s*(\d+)\s*$", text, re.MULTILINE)
        if mt:
            m[key] = int(mt.group(1))

    # Per-core frontend cycles
    m["per_core"] = {}
    for key in ["memory_access_cycles_recorded_core", "cycles_recorded_core"]:
        per_core = []
        for mt in re.finditer(rf"^\s*{key}_(\d+):\s*(\d+)\s*$", text, re.MULTILINE):
            ci, v = int(mt.group(1)), int(mt.group(2))
            _ensure_len(per_core, ci, 0)
            per_core[ci] = v
        m["per_core"][key] = per_core

    for key in ["llc_read_access", "llc_read_misses", "llc_write_access", "llc_write_misses"]:
        mt = re.search(rf"^\s*{key}:\s*(\d+)\s*$", text, re.MULTILINE)
        if mt:
            m[key] = int(mt.group(1))

    return m


# =============================================================================
# Derived metrics
# =============================================================================

def _arr_get(arr: list, i: int, default=0):
    return arr[i] if i < len(arr) else default


def per_channel_row_sum(m: dict, ch: int) -> int:
    """Per-channel read count via row counters (divergence sentinel only)."""
    return (
        _arr_get(m["read_row_hits_per_channel"], ch)
        + _arr_get(m["read_row_misses_per_channel"], ch)
        + _arr_get(m["read_row_conflicts_per_channel"], ch)
    )


def per_channel_avg_lat(m: dict, ch: int) -> float:
    cnt = _arr_get(m["num_read_reqs_per_channel"], ch)
    if cnt == 0:
        return math.nan
    return _arr_get(m["read_latency_per_channel"], ch) / cnt


def summarize(metrics: dict, expected_cores: int, per_core_insts: int) -> dict:
    counts = list(metrics["num_read_reqs_per_channel"])
    total_channels = max(len(counts), len(metrics["read_row_hits_per_channel"]))
    active_idx = [i for i in range(total_channels) if _arr_get(counts, i) > 0]
    n_active = len(active_idx)

    # Sentinel: in Ramulator 2.1 the native read count agrees with the row-counter
    # sum (they differ only by read-forwarded reqs that bypass the row machinery).
    # A large divergence would mean the old 2.0 num_read_reqs inflation is back.
    nrr_total = sum(counts)
    row_total = sum(per_channel_row_sum(metrics, ch) for ch in range(total_channels))
    if nrr_total and abs(nrr_total - row_total) / nrr_total > 0.05:
        print(f"    [WARN] num_read_reqs ({nrr_total:,}) diverges from row-counter "
              f"sum ({row_total:,}) by >5% — counter regression?", flush=True)

    # Aggregate read-row stats
    hits = sum(metrics["read_row_hits_per_channel"])
    misses = sum(metrics["read_row_misses_per_channel"])
    conflicts = sum(metrics["read_row_conflicts_per_channel"])
    total_acc = hits + misses + conflicts
    row_hit_rate = hits / total_acc if total_acc > 0 else math.nan

    # Native per-channel read latency, averaged across active channels.
    sys_cyc = metrics["memory_system_cycles"] or 0
    lats = [per_channel_avg_lat(metrics, ch) for ch in active_idx]
    lats = [x for x in lats if not math.isnan(x)]
    avg_lat = (sum(lats) / len(lats)) if lats else math.nan

    # Read queue length, averaged across active channels.
    rq = []
    for ch in active_idx:
        v = _arr_get(metrics["read_queue_len_avg_per_channel"], ch, math.nan)
        if not math.isnan(v):
            rq.append(v)
    avg_rq = (sum(rq) / len(rq)) if rq else math.nan

    # Per-core runtime: max(cycles_recorded_core_i) is the slowest core's
    # finishing time, i.e. wall-clock simulation length at the frontend.
    cyc_per_core = metrics["per_core"].get("cycles_recorded_core", [])
    mac_per_core = metrics["per_core"].get("memory_access_cycles_recorded_core", [])
    runtime_cycles = max(cyc_per_core) if cyc_per_core else None

    # IPC = retired instructions / runtime cycles. With multi-core, each core
    # retires per_core_insts. Aggregate system throughput in IPS = n_cores * IPC.
    if runtime_cycles and per_core_insts:
        per_core_ipc = per_core_insts / runtime_cycles
    else:
        per_core_ipc = math.nan

    # The avg_mem_stall_fraction metric DOES NOT capture MSHR/queue-full stalls;
    # in saturated regimes it can decrease counterintuitively. Reported here for
    # cross-reference only; primary throughput signal is per_core_ipc.
    stall_fracs = [
        (mac_per_core[i] / cyc_per_core[i])
        for i in range(min(len(mac_per_core), len(cyc_per_core)))
        if cyc_per_core[i] > 0
    ]
    avg_stall = sum(stall_fracs) / len(stall_fracs) if stall_fracs else math.nan

    return {
        "memory_system_cycles": sys_cyc,
        "runtime_cycles": runtime_cycles,
        "per_core_ipc": per_core_ipc,
        "n_cores_observed": len(cyc_per_core),
        "n_cores_expected": expected_cores,
        "active_channels": n_active,
        "total_channels": total_channels,
        "avg_read_latency": avg_lat,
        "avg_read_queue_len": avg_rq,
        "row_hit_rate": row_hit_rate,
        "total_read_reqs": metrics["total_num_read_requests"],
        "avg_mem_stall_fraction_incomplete": avg_stall,
    }


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--gen-script", type=Path, default=Path("gen/gen_kv_decode_trace.py"))
    p.add_argument("--out-dir", type=Path, default=Path("results/core_batch_sweep"))
    p.add_argument("--trace-dir", type=Path, default=Path("traces/core_batch_sweep"))
    p.add_argument("--cores", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32],
                   help="Concurrent-core counts to sweep (one independent decode "
                        "stream per core). The concurrency / contention axis.")
    p.add_argument("--batch", type=int, nargs="+", default=[1],
                   help="Per-trace LLM batch dims to sweep (serial within a core). "
                        "Raises per-core footprint, not concurrency. Full grid = "
                        "batch x cores.")

    p.add_argument("--num-layers", type=int, default=32)
    p.add_argument("--num-query-heads", type=int, default=32)
    p.add_argument("--num-kv-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--context-len", type=int, default=512)
    p.add_argument("--decode-tokens", type=int, default=1)
    p.add_argument("--inst-slack", type=float, default=1.1)
    p.add_argument("--gqa-emission", choices=["reread", "naive"], default=None,
                   help="Forwarded to the generator (default: generator's own "
                        "default, reread). 'naive' regenerates the legacy "
                        "emission under otherwise-identical settings for 1:1 "
                        "mode comparison.")

    p.add_argument("--base-k", type=lambda s: int(s, 0), default=DEFAULT_BASE_K,
                   help="Base virtual address for core 0's K region. V base and "
                        "the per-core region size are derived automatically.")
    p.add_argument("--core-region-size", type=lambda s: int(s, 0), default=None,
                   help="Override per-core (K and V) region size. Default: auto, "
                        "sized to the per-core tensor (>= it, else cores alias).")

    p.add_argument("--dram", nargs="+", choices=sorted(ram21.MEMORIES),
                   default=["ddr4"],
                   help="DRAM configs to run per grid point. Default: ddr4 only.")
    p.add_argument("--llc-associativity", default=None,
                   help="LLC ways: an int, or 'auto' = sim_constants."
                        "llc_auto_assoc(n) — the minimal ways keeping the set "
                        "count pow2 and >= 6 (odd part of n doubled to >= 6; "
                        "pow2 n -> 8 = the default). Guards against both "
                        "SimpleO3's non-pow2-set mis-indexing (false hits at "
                        "9/10/11 cores with default 8 ways) and the low-assoc "
                        "conflict-miss seam (real-capture n3 at 3-way: +3.5% "
                        "runtime; clean by 6-way).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.gen_script.exists():
        sys.stderr.write(f"ERROR: not found: {args.gen_script}\n"); sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.trace_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    # ---- CSV (rewritten after every completed run, so a hard fail mid-sweep
    # keeps all rows completed so far) ----
    # Only domain-safe metrics. Dropped on purpose:
    #   memory_system_cycles, raw avg_read_latency (cycles) -> memory-clock domain;
    #     a memory-clock-domain quantity. Use runtime_cycles (CPU domain) and
    #     avg_read_latency_ns instead.
    #   avg_mem_stall_fraction_incomplete -> measurement is incomplete and
    #     contradicts IPC under saturation; misleading, so omitted.
    cols = ["batch", "cores", "dram",
            "runtime_cycles", "per_core_ipc",
            "n_cores_observed", "n_cores_expected",
            "active_channels", "total_channels",
            "avg_read_latency_ns", "avg_read_queue_len", "row_hit_rate",
            "total_read_reqs", "read_bw_gbs", "power", "llc_assoc"]
    csv_path = args.out_dir / "core_batch_sweep_summary.csv"

    def write_csv() -> None:
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in cols})

    for batch in args.batch:
      for n_cores in args.cores:
        print(f"\n========== batch={batch} cores={n_cores} ==========", flush=True)

        try:
            trace_paths, per_core_insts, region = generate_traces(n_cores, batch, args)
        except Exception as e:
            sys.stderr.write(
                f"ERROR: trace generation failed (batch={batch}, cores={n_cores}): {e}\n"
            )
            sys.exit(1)

        num_expected = int(per_core_insts * args.inst_slack)
        if num_expected > MAX_EXPECTED_INSTS:
            sys.stderr.write(
                f"ERROR: batch={batch} cores={n_cores}: per-core insts {per_core_insts} "
                f"* slack = {num_expected} exceeds SimpleO3 cap {MAX_EXPECTED_INSTS}. "
                f"Reduce --context-len/--batch.\n"
            )
            sys.exit(1)

        print(f"  generated {len(trace_paths)} cores, region={region/2**20:.0f}MiB, "
              f"per_core_insts={per_core_insts:,}, num_expected_insts={num_expected:,}")

        for dram_name in args.dram:
            stats_path = args.out_dir / f"{dram_name}_b{batch}_n{n_cores}.stats"

            llc_assoc = None
            if args.llc_associativity is not None:
                llc_assoc = (llc_auto_assoc(n_cores)
                             if args.llc_associativity == "auto"
                             else int(args.llc_associativity))

            print(f"  running {dram_name} "
                  f"{'' if llc_assoc is None else f'(llc_assoc={llc_assoc}) '}...",
                  flush=True)
            try:
                sim = ram21.build_sim(
                    dram_name, trace_paths, num_expected_insts=num_expected,
                    llc_associativity=llc_assoc,
                )
                ram21.run_to_stats_file(sim, stats_path)
            except Exception as e:  # noqa: BLE001
                sys.stderr.write(
                    f"ERROR: run failed ({dram_name}, batch={batch}, cores={n_cores}): "
                    f"{e}; see {stats_path}\n"
                )
                sys.exit(1)

            metrics = parse_stats(stats_path)
            s = summarize(metrics, expected_cores=n_cores, per_core_insts=per_core_insts)
            s["batch"] = batch
            s["cores"] = n_cores
            s["dram"] = dram_name
            # Convert the memory-cycle latency to ns (domain-neutral, via the
            # standard's time_unit_ns).
            tu = ram21.MEMORIES[dram_name]["time_unit_ns"]
            s["avg_read_latency_ns"] = (
                s["avg_read_latency"] * tu
                if not math.isnan(s["avg_read_latency"]) else math.nan
            )
            # Total read bandwidth. bytes are bytes and runtime_cycles is the
            # CPU domain (3.2 GHz).
            s["read_bw_gbs"] = (
                s["total_read_reqs"] * ram21.CACHE_LINE
                / (s["runtime_cycles"] / CPU_HZ) / 1e9
                if s["runtime_cycles"] and s["total_read_reqs"] is not None
                else math.nan
            )
            # Kleinrock power P = throughput/latency ((GB/s)/ns; relative use
            # only — the argmax over a sweep locates the knee, the absolute
            # value means nothing).
            s["power"] = (
                s["read_bw_gbs"] / s["avg_read_latency_ns"]
                if not (math.isnan(s["read_bw_gbs"]) or math.isnan(s["avg_read_latency_ns"]))
                and s["avg_read_latency_ns"] else math.nan
            )
            # Effective LLC ways (pow2-set seam is assoc-dependent; record it).
            s["llc_assoc"] = llc_assoc if llc_assoc is not None else LLC_DEFAULT_ASSOC
            rows.append(s)
            write_csv()

            print(
                f"    sys_cyc={s['memory_system_cycles']:>14,}  "
                f"runtime={s['runtime_cycles']:>14,}  "
                f"IPC={s['per_core_ipc']:.3f}  "
                f"cores={s['n_cores_observed']}/{s['n_cores_expected']}  "
                f"act_ch={s['active_channels']}/{s['total_channels']}  "
                f"lat={s['avg_read_latency']:.1f}c  "
                f"rd_qlen={s['avg_read_queue_len']:.2f}  "
                f"row_hit={s['row_hit_rate']:.4f}"
            )

    if not rows:
        sys.stderr.write("\nNo successful runs.\n"); sys.exit(1)

    print(f"\nSummary CSV: {csv_path}")


if __name__ == "__main__":
    main()