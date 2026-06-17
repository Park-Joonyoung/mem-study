# sim_raw — Ramulator SimpleO3 stats backing the sim side of the hw-vs-sim comparison

These `canon.<size>.<assoc>.stats` are the raw simulator outputs for the LLC
cache-size sweep summarized in `../sim_llc_size_sweep.csv`. They were produced by
running the **real canonical study trace** through SimpleO3 at varying LLC sizes:

    trace = traces/compute_sweep/kv_b1_n1_c512_t1_L32_comp32.trace   (n_kv=512, decode-tokens=1)

The sweep shows the sim's per-cycle DRAM demand is **flat at ~0.986 bytes/cycle
from 2 MiB to 32 MiB LLC**, then drops to the 64 MiB unique footprint only at
**≥64 MiB** — proving the LLC size (incl. matching it to the host's 27.5 MiB L3)
does not change the sim demand; the hw/sim gap is the compute model.

## Reproduce a point
Generate the canonical decode trace (1 decode step, GQA reread default):

    python gen/gen_kv_decode_trace.py \
        --num-layers 32 --num-query-heads 32 --num-kv-heads 8 --head-dim 128 \
        --context-len <n_kv> --decode-tokens 1 --batch 1 \
        --compute-per-k-line 32 --compute-per-v-line 32 --out <trace>

Simulate (run/ram21.py):

    build_sim('ddr4', [trace], num_expected_insts=int(insts*1.1),
              llc_associativity=<A>, llc_capacity=<SIZE>)

LLC params used (set_size = size/(64·ways) MUST be a power of two):
  - 2/4/8/16/32/64/128 MB  -> ways=8     (pow2 sizes -> pow2 set count)
  - 27.5 MiB               -> "28160KB", ways=55  -> 450560/55 = 8192 sets (pow2)

NOTE: parse_capacity_str reads only the leading integer ("27.5MB" -> 27 MiB);
use "28160KB" for exactly 27.5 MiB. A non-pow2 set count mis-indexes
catastrophically (llc.cpp index_mask = set_size-1).
