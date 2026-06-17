#!/usr/bin/env python3
"""Expand a GGML_KV_LOG metadata log into SimpleO3 per-core traces.

Input: the per-op metadata lines written by the local llama.cpp KV-trace
instrumentation (ggml-cpu.c / ops.cpp, build-kvtrace):

  MM root=cache_k_l0 data=0x7f.. ne=128,512,8,1 nb=2,2048,256,1048576 n_tok=1 type=f16
  SR root=cache_v_l0 data=0x7f.. ne=1,262144,1,1 nb=2,2,524288,524288 \
     n_idx=1024 idx=511,1023,1535,2047 idx_last=524287 type=f16

This script replays the exact CPU-side access order and emits one SimpleO3
trace line per touched cache line:

  MM (mul_mat reading a KV view as src0, FA disabled): replicates
      ggml_compute_forward_mul_mat's chunking for a SINGLE-thread run
      (nth=1, GGML_LLAMAFILE=OFF, non-NUMA: chunk_size 16, chunks processed
      in index order) and one_chunk's 16x16 block loops; each vec_dot reads
      src0 row [data + i02*nb02 + ir0*nb01, +ne00*tsize). f16 nrows==1.
  SR (ggml_set_rows writing new K/V): row indices at decode time are an
      arithmetic sequence (verified against the logged first-4 + last
      samples); row i1 writes [data + i1*nb1, +ne0*tsize) -> write-allocate.

Only DECODE passes (MM n_tok == 1) are expanded; warmup/prefill passes are
skipped. The pass structure must be complete: n_layers x (SR k, SR v, MM k,
MM v).

Trace-level conventions follow gen_kv_decode_trace.py so real and synthetic
traces differ ONLY in address pattern and stream order:
  read line  = "<compute> <line>"            compute = 32 per line PER Q-HEAD
  write line = "<compute> <line> <line>"     compute = 2 (write-allocate)
Note: the real mul_mat streams each KV row once per query head (GQA group
re-reads; LLC absorbs them), where the synthetic trace emits each line once
with group_size-amplified compute. Total compute matches; the real trace has
~group_size x more memory instructions whose repeats are LLC hits.

All addresses from the single captured process are rebased into disjoint
per-core regions (real byte offsets within the KV arena are preserved):
  core i: base + i * region,  region = align_up(arena_span, 1 MiB)

Outputs <tag>_b1_core<i>_c<kv_size>_t<passes>_L<layers>.trace plus meta.json
(per-core inst count, region size, file list) for the sweep runner.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_utils import (  # noqa: E402
    CACHE_LINE, COMPUTE_PER_KV_LINE, COMPUTE_PER_STORE_LINE, DEFAULT_BASE_K,
    ceil_div,
)

TYPE_SIZE = {"f16": 2, "bf16": 2, "f32": 4}

LINE_RE = re.compile(
    r"^(?P<tag>MM|SR) root=(?P<root>\S+) data=0x(?P<data>[0-9a-fA-F]+) "
    r"ne=(?P<ne>[\d,]+) nb=(?P<nb>[\d,]+) "
    r"(?:n_tok=(?P<n_tok>-?\d+)|n_idx=(?P<n_idx>\d+) idx=(?P<idx>[-\d,]+) idx_last=(?P<idx_last>-?\d+)) "
    r"type=(?P<type>\S+)$"
)

ROOT_RE = re.compile(r"^cache_(?P<kv>[kv])_l(?P<layer>\d+)$")


class Op:
    __slots__ = ("tag", "kv", "layer", "data", "ne", "nb", "n_tok",
                 "n_idx", "idx", "idx_last", "tsize", "lineno")

    def __init__(self, m: re.Match, lineno: int):
        self.tag = m["tag"]
        rm = ROOT_RE.match(m["root"])
        if not rm:
            raise ValueError(f"line {lineno}: unexpected root {m['root']!r}")
        self.kv = rm["kv"]
        self.layer = int(rm["layer"])
        self.data = int(m["data"], 16)
        self.ne = [int(x) for x in m["ne"].split(",")]
        self.nb = [int(x) for x in m["nb"].split(",")]
        self.n_tok = int(m["n_tok"]) if m["n_tok"] is not None else None
        self.n_idx = int(m["n_idx"]) if m["n_idx"] is not None else None
        self.idx = [int(x) for x in m["idx"].split(",")] if m["idx"] else None
        self.idx_last = int(m["idx_last"]) if m["idx_last"] is not None else None
        if m["type"] not in TYPE_SIZE:
            raise ValueError(f"line {lineno}: unsupported type {m['type']}")
        self.tsize = TYPE_SIZE[m["type"]]
        self.lineno = lineno


def parse_log(path: Path) -> list[Op]:
    ops = []
    with path.open() as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            m = LINE_RE.match(line)
            if not m:
                raise ValueError(f"line {lineno}: unparseable: {line[:120]}")
            ops.append(Op(m, lineno))
    return ops


def split_passes(ops: list[Op]) -> list[list[Op]]:
    """A new forward pass starts whenever the layer index decreases."""
    passes: list[list[Op]] = []
    prev_layer = None
    for op in ops:
        if prev_layer is None or op.layer < prev_layer:
            passes.append([])
        passes[-1].append(op)
        prev_layer = op.layer
    return passes


def is_decode_pass(ops: list[Op]) -> bool:
    mm = [op for op in ops if op.tag == "MM"]
    return bool(mm) and all(op.n_tok == 1 for op in mm)


def validate_decode_pass(ops: list[Op]) -> dict:
    """Check the n_layers x (SR k, SR v, MM k, MM v) structure; return facts."""
    layers = sorted({op.layer for op in ops})
    n_layers = len(layers)
    if layers != list(range(n_layers)):
        raise ValueError(f"non-contiguous layer set: {layers}")
    by_kind = {}
    for op in ops:
        by_kind.setdefault((op.tag, op.kv), []).append(op)
    for kind in [("SR", "k"), ("SR", "v"), ("MM", "k"), ("MM", "v")]:
        got = len(by_kind.get(kind, []))
        if got != n_layers:
            raise ValueError(f"decode pass: expected {n_layers} {kind} ops, got {got}")

    # kv_size from the K cache root: SR k dst is the [n_embd_gqa, kv_size] root view.
    srk = by_kind[("SR", "k")][0]
    kv_size = srk.ne[1]
    mmk = by_kind[("MM", "k")][0]
    n_kv = mmk.ne[1]
    if n_kv != kv_size:
        print(f"  [note] MM K view covers n_kv={n_kv} of kv_size={kv_size} "
              f"(padding rounds to 256)", file=sys.stderr)
    return {"n_layers": n_layers, "kv_size": kv_size, "n_kv": n_kv}


# =============================================================================
# Access-stream expansion (offsets relative to the captured address space)
# =============================================================================

def emit_row(out: list, addr: int, length: int, compute: int, write: bool) -> None:
    first = addr & ~(CACHE_LINE - 1)
    last = (addr + length - 1) & ~(CACHE_LINE - 1)
    for line in range(first, last + CACHE_LINE, CACHE_LINE):
        out.append((compute, line, write))


def expand_mm(op: Op, n_q_heads: int, compute_read: int, out: list) -> None:
    """ggml_compute_forward_mul_mat, nth=1, GGML_LLAMAFILE=OFF, f16 nrows=1."""
    ne00, ne01, ne02, ne03 = op.ne
    nb00, nb01, nb02, nb03 = op.nb
    n_tok = op.n_tok

    # dst dims (kq: [n_kv, n_tok, n_heads]; kqv: [head_dim, n_tok, n_heads])
    ne1, ne2, ne3 = n_tok, n_q_heads, 1
    ne12 = n_q_heads
    r2 = ne12 // ne02
    nr0 = ne01
    nr1 = ne1 * ne2 * ne3

    chunk_size = 64 if (nr0 == 1 or nr1 == 1) else 16
    nchunk0 = ceil_div(nr0, chunk_size)
    nchunk1 = ceil_div(nr1, chunk_size)
    nth = 1  # single-thread capture; chunks are then processed in index order
    if nchunk0 * nchunk1 < nth * 4:
        nchunk0 = nth if nr0 > nr1 else 1
        nchunk1 = 1 if nr0 > nr1 else nth
    dr0 = ceil_div(nr0, nchunk0)
    dr1 = ceil_div(nr1, nchunk1)

    row_bytes = ne00 * op.tsize
    blck = 16
    for chunk in range(nchunk0 * nchunk1):
        ith0 = chunk % nchunk0
        ith1 = chunk // nchunk0
        ir0_start, ir0_end = dr0 * ith0, min(dr0 * ith0 + dr0, nr0)
        ir1_start, ir1_end = dr1 * ith1, min(dr1 * ith1 + dr1, nr1)
        if ir0_start >= ir0_end or ir1_start >= ir1_end:
            continue
        for iir1 in range(ir1_start, ir1_end, blck):
            for iir0 in range(ir0_start, ir0_end, blck):
                for ir1 in range(iir1, min(iir1 + blck, ir1_end)):
                    i13 = ir1 // (ne12 * ne1)
                    i12 = (ir1 - i13 * ne12 * ne1) // ne1
                    i02 = i12 // r2
                    i03 = i13
                    row_base = op.data + i02 * nb02 + i03 * nb03
                    for ir0 in range(iir0, min(iir0 + blck, ir0_end)):
                        emit_row(out, row_base + ir0 * nb01, row_bytes,
                                 compute_read, write=False)


def expand_sr(op: Op, compute_store: int, out: list) -> None:
    """ggml_set_rows: write rows idx[i] in ascending i (single thread)."""
    n_idx = op.n_idx
    samples = [v for v in op.idx if v >= 0][: min(4, n_idx)]
    stride = samples[1] - samples[0] if n_idx > 1 else 0
    for a, b in zip(samples, samples[1:]):
        if b - a != stride:
            raise ValueError(
                f"line {op.lineno}: SR idx samples {samples} are not an "
                f"arithmetic sequence; cannot reconstruct the store stream")
    expect_last = samples[0] + (n_idx - 1) * stride
    if op.idx_last != expect_last:
        raise ValueError(
            f"line {op.lineno}: SR idx_last={op.idx_last} != extrapolated "
            f"{expect_last} (start={samples[0]}, stride={stride}, n={n_idx})")
    row_bytes = op.ne[0] * op.tsize
    nb1 = op.nb[1]
    for i in range(n_idx):
        emit_row(out, op.data + (samples[0] + i * stride) * nb1, row_bytes,
                 compute_store, write=True)


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", type=Path, required=True, help="GGML_KV_LOG file")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--tag", default="real")
    p.add_argument("--cores", type=int, default=8,
                   help="Number of rebased per-core trace copies to write.")
    p.add_argument("--num-query-heads", type=int, default=32,
                   help="src1 head count of the attention mul_mats (not in the "
                        "log; needed for GQA broadcast r2 = q_heads/kv_heads).")
    p.add_argument("--passes", type=int, default=None,
                   help="Expand this many decode passes (default: all found).")
    p.add_argument("--base", type=lambda s: int(s, 0), default=DEFAULT_BASE_K)
    p.add_argument("--compute-read", type=int, default=COMPUTE_PER_KV_LINE,
                   help="Non-mem insts per read line (per q-head stream; "
                        "single-sourced with the synthetic generator via "
                        "sim_constants.py).")
    p.add_argument("--compute-store", type=int, default=COMPUTE_PER_STORE_LINE)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ops = parse_log(args.log)
    passes = split_passes(ops)
    decode_passes = [p for p in passes if is_decode_pass(p)]
    print(f"log: {len(ops)} ops, {len(passes)} passes "
          f"({len(decode_passes)} decode)")
    if not decode_passes:
        raise SystemExit("no decode (n_tok==1) passes found")
    if args.passes is not None:
        decode_passes = decode_passes[: args.passes]

    facts = validate_decode_pass(decode_passes[0])
    for dp in decode_passes[1:]:
        validate_decode_pass(dp)

    # Expand all selected decode passes into one serial stream (true offsets).
    accesses: list = []
    for dp in decode_passes:
        for op in dp:
            if op.tag == "MM":
                expand_mm(op, args.num_query_heads, args.compute_read, accesses)
            else:
                expand_sr(op, args.compute_store, accesses)

    lo = min(a[1] for a in accesses)
    hi = max(a[1] for a in accesses) + CACHE_LINE
    lo &= ~(CACHE_LINE - 1)
    span = hi - lo
    region = ceil_div(span, 1 << 20) * (1 << 20)

    n_reads = sum(1 for a in accesses if not a[2])
    n_writes = len(accesses) - n_reads
    insts = sum(c + 1 for c, _, _ in accesses)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for core in range(args.cores):
        delta = args.base + core * region - lo
        path = args.out_dir / (
            f"{args.tag}_b1_core{core}_c{facts['kv_size']}"
            f"_t{len(decode_passes)}_L{facts['n_layers']}.trace")
        with path.open("w") as f:
            for compute, line, write in accesses:
                addr = line + delta
                if write:
                    f.write(f"{compute} {addr} {addr}\n")
                else:
                    f.write(f"{compute} {addr}\n")
        files.append(str(path))
        print(f"wrote {path}")

    meta = {
        "tag": args.tag,
        "log": str(args.log),
        "decode_passes": len(decode_passes),
        "n_layers": facts["n_layers"],
        "kv_size": facts["kv_size"],
        "n_kv": facts["n_kv"],
        "num_query_heads": args.num_query_heads,
        "arena_span_bytes": span,
        "region_bytes": region,
        "base": args.base,
        "trace_lines": len(accesses),
        "reads": n_reads,
        "writes": n_writes,
        "per_core_insts": insts,
        "files": files,
    }
    meta_path = args.out_dir / f"{args.tag}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {meta_path}")
    print(f"arena span={span / 2**20:.2f} MiB, region={region / 2**20:.0f} MiB, "
          f"lines={len(accesses):,} (reads={n_reads:,} writes={n_writes:,})")
    print(f"estimated retired instructions for one pass={insts}")


if __name__ == "__main__":
    main()
