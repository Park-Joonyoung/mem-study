#!/usr/bin/env python3
"""Generate a SimpleO3 trace for LLM decode-time KV-cache traffic.

The model is intentionally trace-level and memory-focused:
  1. For each generated token, write the new K/V cache line(s).
  2. Stream all previous K cache lines for attention score computation.
  3. Stream all previous V cache lines for value aggregation.

This is not a full transformer simulator. It is meant to expose the memory
behavior of decode attention, especially long sequential reads from KV cache.

GQA (Grouped Query Attention) modeling
--------------------------------------
We assume a fused-attention kernel (FlashAttention / vLLM / llama.cpp style):
each KV cache line is fetched from DRAM once per KV head and reused across
the group_size = num_query_heads / num_kv_heads query heads that share that
KV head. How that reuse appears in the trace is controlled by --gqa-emission:

reread (default since 2026-06-11):
    Walk the sequence in --reread-block-positions blocks (default 16 = the
    row-block size blck_0 of llama.cpp's mul_mat one_chunk kernel); each
    block is swept group_size times -- one pass per query head -- and every
    pass emits every line of the block with --compute-per-{k,v}-line (the
    per-q-head cost) attached. Pass 1 of a block is a run of DRAM misses;
    passes 2..group_size re-walk the same few KB and hit the LLC by
    construction.

naive (legacy behavior until 2026-06-11):
    Each line emitted exactly once, compute amplified by group_size.

Both modes produce identical DRAM-visible traffic (same distinct lines; the
re-reads are LLC hits) and identical total compute. They differ only in how
that work is laid out along the instruction stream -- which turned out to
dominate the multi-core contention behavior this study measures.

Why the default changed (validated against an instrumented llama.cpp capture,
fa0/c512/t1 single-thread decode; see gen/expand_kv_log.py and
results/real_core_sweep/real_vs_synthetic_b1.csv):

1. Naive emission serializes misses. A trace line "<compute> <addr>"
   makes SimpleO3 retire ~32*group_size = 128 filler instructions between
   consecutive loads, and nearly every load is a DRAM miss. With a reorder
   window of the same order, at most ~one miss is in flight: MLP ~= 1,
   single-core demand 1.9 GB/s, and the core sweep showed almost no
   contention on a 19.2 GB/s DDR4-2400 channel even at 8 cores
   (runtime +6% vs n1, latency flat at ~21.8 ns, read queue ~1).

2. The real kernel is bursty, not uniform. llama.cpp's mul_mat walks a
   16-position row block once per query head: the group's first pass is a
   back-to-back miss run (~33 insts apart, several misses in the O3 window
   simultaneously), the next group_size-1 passes are fast LLC-hit re-walks.
   Same AVERAGE spacing (~132 insts/miss), completely different timing:
   measured per-core demand is 3.1 GB/s, 1.6x naive, so 8 real streams
   oversubscribe the channel (8 x 3.1 > 19.2 GB/s). The real-trace sweep
   saturates -- latency 23.5 -> 76.4 ns, read queue 13.8, runtime +68% at
   n8, knee visible from n4 -- none of which the naive trace showed.
   Reread reproduces the miss-burst/hit-gap structure that causes this.

3. Re-reads make the shared LLC a real resource. With one-shot emission the
   LLC never sees reuse, so inter-core LLC contention cannot exist in the
   model. With reread, group reuse must actually hit; under multi-core
   pressure some of it does not (the real trace loses ~3.6% of re-reads to
   conflict misses at 8 cores). That effect is now representable.

4. Instruction counts line up with the real stream (b1/c512/t1/L32: 138.7M
   synthetic-reread vs 138.5M captured), keeping runtime_cycles an
   apples-to-apples metric between synthetic and real traces.

Caveats: a reread trace is ~group_size x larger (lines and insts) than
naive; sweep tooling reads the printed inst count, so nothing else needs
adjusting. Result sets produced before 2026-06-11 (results/core_sweep_b1,
results/batch_sweep_n8) used naive emission -- regenerate, or pass
--gqa-emission naive when extending them. Address layout and loop order
(real token-major K, transposed V with scattered appends) are intentionally
NOT changed here; those are separate, later fidelity steps.

Reference configurations (Llama family, all fp16, head_dim=128 unless noted)
  Llama-3.1-8B:  layers=32, q_heads=32, kv_heads=8  (group=4)
  Llama-3.1-70B: layers=80, q_heads=64, kv_heads=8  (group=8)
  Llama-3.2-1B:  layers=16, q_heads=32, kv_heads=8, head_dim=64  (group=4)
  Llama-3.2-3B:  layers=28, q_heads=24, kv_heads=8  (group=3)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from trace_utils import (
    CACHE_LINE, COMPUTE_PER_KV_LINE, COMPUTE_PER_STORE_LINE,
    DEFAULT_BASE_K, DEFAULT_BASE_V, TRACE_DIR,
    ceil_div, emit_read, emit_write_alloc, ensure_trace_dir,
)


# Defaults are modest enough for Ramulator2 to load quickly, but large enough
# to exceed the default 2MB LLC. Defaults track Llama-3.1-8B head config but
# with fewer layers for fast iteration.
DEFAULT_NUM_LAYERS = 8
DEFAULT_BATCH = 1
DEFAULT_NUM_QUERY_HEADS = 32
DEFAULT_NUM_KV_HEADS = 8
DEFAULT_HEAD_DIM = 128
DEFAULT_CONTEXT_LEN = 2048
DEFAULT_DECODE_TOKENS = 4
DEFAULT_DTYPE_BYTES = 2

# 단일 출처: sim_constants.py (expander와 공유하는 real<->synthetic 계약 —
# 한쪽만 바꾸면 inst 수/miss 간격이 어긋나 비교가 조용히 깨진다)
DEFAULT_COMPUTE_PER_K_LINE = COMPUTE_PER_KV_LINE    # SIMD가 아닌 일반 MAC의 경우 32로 고정
DEFAULT_COMPUTE_PER_V_LINE = COMPUTE_PER_KV_LINE    # SIMD가 아닌 일반 MAC의 경우 32로 고정
DEFAULT_COMPUTE_PER_STORE_LINE = COMPUTE_PER_STORE_LINE

# Virtual address layout (단일 출처: sim_constants.py).
BASE_K = DEFAULT_BASE_K
BASE_V = DEFAULT_BASE_V

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--num-layers", type=int, default=DEFAULT_NUM_LAYERS)       # 레이어 수
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)                 # 배치 수
    parser.add_argument(                                                            # Q 헤드 수
        "--num-query-heads",
        type=int,
        default=DEFAULT_NUM_QUERY_HEADS,
        help="Total query heads. With GQA, must be a multiple of --num-kv-heads.",
    )
    parser.add_argument("--num-kv-heads", type=int, default=DEFAULT_NUM_KV_HEADS)   # KV 헤드 수
    parser.add_argument("--head-dim", type=int, default=DEFAULT_HEAD_DIM)           # d_q, d_k, d_v
    parser.add_argument("--context-len", type=int, default=DEFAULT_CONTEXT_LEN)     # context 길이 | decode 시작 전에 이미 쌓여 있는 토큰 수 (prefill)
    parser.add_argument("--decode-tokens", type=int, default=DEFAULT_DECODE_TOKENS) # 새로 생성되는 토큰의 수
    parser.add_argument("--dtype-bytes", type=int, default=DEFAULT_DTYPE_BYTES)     # precision | Default : 2 Bytes = 16 bits
    parser.add_argument(                                                            # 토큰이 한 개 생성될 때 K cache line 하나 당 Q 헤드 1개의 연산의 수 (QK^T)
        "--compute-per-k-line",
        type=int,
        default=DEFAULT_COMPUTE_PER_K_LINE,
        help="Compute ops per K cache line load, per query head. In 'reread' "
             "mode attached to each per-q-head emission; in 'naive' mode "
             "scaled by group_size onto the single emission.",
    )
    parser.add_argument(                                                            # 토큰이 한 개 생성될 때 V cache line 하나 당 Q 헤드 1개의 연산의 수 (AV)
        "--compute-per-v-line",
        type=int,
        default=DEFAULT_COMPUTE_PER_V_LINE,
        help="Compute ops per V cache line load, per query head. In 'reread' "
             "mode attached to each per-q-head emission; in 'naive' mode "
             "scaled by group_size onto the single emission.",
    )
    parser.add_argument(                                                            # GQA 재사용을 트레이스에 표현하는 방식 (docstring 근거 참고)
        "--gqa-emission",
        choices=["reread", "naive"],
        default="reread",
        help="'reread' (default): emit each KV line once per query head, in "
             "--reread-block-positions blocks -- bursty miss runs + LLC-hit "
             "re-walks, matching measured llama.cpp timing behavior. "
             "'naive': pre-2026-06-11 behavior, one emission per line with "
             "compute x group_size. See the module docstring for why.",
    )
    parser.add_argument(                                                            # 재독 블록의 seq 길이 = mul_mat one_chunk의 row block(blck_0)
        "--reread-block-positions",
        type=int,
        default=16,
        help="Sequence-block size for reread mode; 16 matches llama.cpp "
             "mul_mat one_chunk's row blocking (blck_0).",
    )
    parser.add_argument("--compute-per-store-line", type=int, default=DEFAULT_COMPUTE_PER_STORE_LINE)   # 토큰이 새로 생성될 때 KV를 append하는 데 드는 연산량
                                                                                                        # projection은 kv decode 트래픽의 범위 밖이므로 작게 설정. 실제로는 매우 큼
    parser.add_argument("--layout", choices=["head_seq_dim", "seq_head_dim"], default="head_seq_dim")   # [head][seq] vs. [seq][head]
    parser.add_argument("--out", type=Path, default=TRACE_DIR / "kv_decode.trace")
    # Per-core base addresses. The defaults match the original single-core layout
    # so existing callers are unaffected. Multi-core callers (e.g. run_kv_core_batch_sweep.py
    # in concurrent mode) pass distinct base addresses per core so that different
    # cores' KV caches occupy disjoint regions of the (virtual) address space.
    parser.add_argument("--base-k", type=lambda s: int(s, 0), default=BASE_K,
                        help="Base virtual address for K cache. Accepts hex (0x...) or decimal.")
    parser.add_argument("--base-v", type=lambda s: int(s, 0), default=BASE_V,
                        help="Base virtual address for V cache. Accepts hex (0x...) or decimal.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    values = [
        args.num_layers,
        args.batch,
        args.num_query_heads,
        args.num_kv_heads,
        args.head_dim,
        args.context_len,
        args.decode_tokens,
        args.dtype_bytes,
        args.compute_per_k_line,
        args.compute_per_v_line,
        args.compute_per_store_line,
        args.reread_block_positions,
    ]
    if min(values) <= 0:
        raise ValueError("shape, dtype size, and compute counts must be positive")
    if args.dtype_bytes > CACHE_LINE:
        raise ValueError("--dtype-bytes must be no larger than the cache line size")
    if args.num_query_heads < args.num_kv_heads:
        raise ValueError("--num-query-heads must be >= --num-kv-heads")
    if args.num_query_heads % args.num_kv_heads != 0:
        raise ValueError(
            "--num-query-heads must be divisible by --num-kv-heads (GQA constraint)"
        )


def kv_addr(
    args: argparse.Namespace,
    max_seq_len: int,
    base: int,
    layer: int,
    batch: int,
    head: int,
    seq: int,
    dim: int,
) -> int:
    if args.layout == "head_seq_dim":   # [layer][batch][head][seq]
        elem = (
            (((layer * args.batch + batch) * args.num_kv_heads + head) * max_seq_len + seq) * args.head_dim
            + dim
        )
    else:                               # [layer][batch][seq][head]
        elem = (
            (((layer * args.batch + batch) * max_seq_len + seq) * args.num_kv_heads + head) * args.head_dim
            + dim
        )
    return base + elem * args.dtype_bytes


def tensor_bytes(args: argparse.Namespace, max_seq_len: int) -> int:    # 전체 레이어/배치의 K 또는 V의 메모리 footprint
    return args.num_layers * args.batch * args.num_kv_heads * max_seq_len * args.head_dim * args.dtype_bytes


def stream_attention_reads(
    f,
    args: argparse.Namespace,
    max_seq_len: int,
    base: int,
    layer: int,
    current_seq: int,
    dim_lines: int,
    elems_per_line: int,
    group_size: int,
    compute_per_line: int,
) -> tuple[int, int]:
    """한 레이어의 K 또는 V 읽기 스트림(seq 0..current_seq)을 emit한다.

    reread: seq를 --reread-block-positions 블록으로 자르고, 블록마다
        group_size번(= 그 kv_head를 공유하는 q-head 패스마다 1번) 재주사.
        첫 패스는 DRAM miss 연속 구간, 나머지 패스는 LLC hit 재주사가 되어
        실측 llama.cpp 커널의 miss-burst/hit-gap 시간 구조를 재현한다
        (근거는 모듈 docstring).
    naive: 라인당 1회, compute에 group_size를 상각 (구버전 동작).

    Returns (estimated_insts, trace_lines).
    """
    insts = 0
    lines = 0
    n_seq = current_seq + 1
    if args.gqa_emission == "reread":
        blk = args.reread_block_positions
        for batch in range(args.batch):
            for head in range(args.num_kv_heads):
                for blk_start in range(0, n_seq, blk):  # mul_mat one_chunk의 16-row 블록에 대응
                    blk_end = min(blk_start + blk, n_seq)
                    for _rep in range(group_size):      # 이 kv_head를 공유하는 q-head 패스들
                        for seq in range(blk_start, blk_end):
                            for dim_line in range(dim_lines):
                                dim = dim_line * elems_per_line
                                insts += emit_read(
                                    f,
                                    compute_per_line,
                                    kv_addr(args, max_seq_len, base, layer, batch, head, seq, dim),
                                )
                                lines += 1
    else:  # naive
        effective = compute_per_line * group_size
        for batch in range(args.batch):
            for head in range(args.num_kv_heads):
                for seq in range(n_seq):
                    for dim_line in range(dim_lines):
                        dim = dim_line * elems_per_line
                        insts += emit_read(
                            f,
                            effective,
                            kv_addr(args, max_seq_len, base, layer, batch, head, seq, dim),
                        )
                        lines += 1
    return insts, lines


def main() -> None:
    args = parse_args()
    validate_args(args)

    ensure_trace_dir()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # GQA: DRAM-visible traffic은 kv_head당 1회로 동일하고, group_size만큼의
    # 재사용을 reread(라인 재독)로 표현할지 naive(compute 상각)로 표현할지는
    # stream_attention_reads()가 --gqa-emission에 따라 결정한다.
    group_size = args.num_query_heads // args.num_kv_heads          # 그룹 사이즈 = Q 헤드 수 / KV 헤드 수

    max_seq_len = args.context_len + args.decode_tokens             # 토큰의 최대 길이 = 기존 토큰 + 새로 생성되는 토큰
    elems_per_line = CACHE_LINE // args.dtype_bytes                 # 한 cache line 당 원소의 개수 = cache line size / precision
    dim_lines = ceil_div(args.head_dim * args.dtype_bytes, CACHE_LINE)  # 한 head의 K/V 벡터(head_dim개 원소) 하나가 차지하는 cache line 수

    trace_lines = 0
    estimated_insts = 0

    with args.out.open("w") as f:
        for token in range(args.decode_tokens):
            current_seq = args.context_len + token      # 현재 토큰의 위치

            for layer in range(args.num_layers):                    # 각 레이어마다~
                # Write new K/V cache lines for this token.
                for batch in range(args.batch):                     # 각 배치마다~
                    for head in range(args.num_kv_heads):           # 각 헤드마다~
                        for dim_line in range(dim_lines):           # 각 헤드 내의 cache line마다~
                            dim = dim_line * elems_per_line             # 해당 line의 시작 원소 위치
                            estimated_insts += emit_write_alloc(        # instruction 추가
                                f,
                                args.compute_per_store_line,
                                kv_addr(args, max_seq_len, args.base_k, layer, batch, head, current_seq, dim),
                            )
                            estimated_insts += emit_write_alloc(
                                f,
                                args.compute_per_store_line,
                                kv_addr(args, max_seq_len, args.base_v, layer, batch, head, current_seq, dim),
                            )
                            trace_lines += 2

                # Stream K cache (이전 토큰 + 현재 토큰까지 전부): GQA 재사용
                # 표현은 --gqa-emission에 따름 (stream_attention_reads 참고).
                ki, kl = stream_attention_reads(
                    f, args, max_seq_len, args.base_k, layer, current_seq,
                    dim_lines, elems_per_line, group_size, args.compute_per_k_line)
                estimated_insts += ki
                trace_lines += kl

                # Stream V cache: same reuse structure as K.
                vi, vl = stream_attention_reads(
                    f, args, max_seq_len, args.base_v, layer, current_seq,
                    dim_lines, elems_per_line, group_size, args.compute_per_v_line)
                estimated_insts += vi
                trace_lines += vl

    k_size = tensor_bytes(args, max_seq_len)    # 전체 K 메모리 footprint 계산
    v_size = tensor_bytes(args, max_seq_len)    # 전체 V 메모리 footprint 계산
    print(f"wrote {args.out}")
    print(f"layout={args.layout}")
    print(
        f"shape layers={args.num_layers} batch={args.batch} "
        f"q_heads={args.num_query_heads} kv_heads={args.num_kv_heads} "
        f"group_size={group_size} head_dim={args.head_dim}"
    )
    print(f"context_len={args.context_len} decode_tokens={args.decode_tokens}")
    if args.gqa_emission == "reread":
        print(
            f"gqa_emission=reread: each line emitted {group_size}x (one pass per "
            f"q-head, {args.reread_block_positions}-position blocks); compute per "
            f"line per pass = K {args.compute_per_k_line} / V {args.compute_per_v_line}"
        )
    else:
        print(
            f"gqa_emission=naive: each line emitted once; compute per line = "
            f"K {args.compute_per_k_line}*{group_size} = {args.compute_per_k_line * group_size} / "
            f"V {args.compute_per_v_line}*{group_size} = {args.compute_per_v_line * group_size}"
        )
    print(f"K range: 0x{args.base_k:x}..0x{args.base_k + k_size - 1:x} ({k_size / 1024 / 1024:.2f} MiB)")
    print(f"V range: 0x{args.base_v:x}..0x{args.base_v + v_size - 1:x} ({v_size / 1024 / 1024:.2f} MiB)")
    print(f"trace lines={trace_lines}")
    print(f"estimated retired instructions for one pass={estimated_insts}")


if __name__ == "__main__":
    main()