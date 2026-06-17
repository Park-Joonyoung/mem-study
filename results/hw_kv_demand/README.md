# hw_kv_demand — hardware-measured per-stream KV-attention DRAM demand

Direct **hardware** measurement of one decode stream's KV-attention DRAM read
demand on the host CPU, as a function of n_kv. This is NOT a simulation:
llama.cpp runs natively and the attention memory traffic is timed/counted with
CPU performance counters.

**Distinct from `results/real_core_sweep/`**, which is the *real captured KV
access trace run through the Ramulator simulator* (real pattern, simulated
execution). Here both the demand AND the timing are measured on silicon —
nothing routes through the sim or the `COMPUTE_PER_KV_LINE=32` constant. The
distinguishing axis is **hardware-vs-simulated**, not real-vs-synthetic.

## Host (incidental measurement machine — not necessarily the study target)
Intel Xeon Gold 5218R — 20 cores/socket, **2.1 GHz base**, **DDR4-2400**,
**L2 1 MiB/core (private)**, **L3 27.5 MiB (shared across all cores)**.

## Method
Instrument `llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c` (env `GGML_KV_TIME=<path>`):
for each decode (n_tok==1) mul_mat whose src0 root is `cache_k*`/`cache_v*`,
bracket the op and accumulate, **per decode token** (token boundary = the
layer-0 K read `cache_k_l0`; n_kv read from the K view's `ne[1]` — V is
transposed so its ne[1] is head_dim, only label from K):
- wall-time via `clock_gettime(MONOTONIC)`
- **actual DRAM bytes** via a per-thread `LLC-read-miss` perf_event
  (`PERF_TYPE_HW_CACHE LL|OP_READ|MISS`, `exclude_kernel`) — isolates THIS
  thread's DRAM reads, no socket-wide IMC contamination from weights/tenants
- logical bytes via `ggml_nelements(src0)*type_size` (for comparison)

Run: `llama-completion -t 1 -tb 32 -fa off -ngl 0 -ctk f16 -ctv f16`,
`models/llama31_8b_random_f16.gguf` (random weights → memory traffic identical
to real weights), n_kv set by prompt length (≈ band_top−30 tokens; real pads
n_kv to 256-multiples). Use `llama-completion`, NOT `llama-cli` (hangs
interactive even with -no-cnv/</dev/null).

**Two controls were essential (each changed the answer):**
1. **CPU frequency pinned to 2.1 GHz base** (`cpupower -g performance` +
   `no_turbo=1`): run-to-run variance **31% → 1.8%** (powersave turbo-ramp was
   the dominant noise; 512×3 = 2.790/2.742/2.775 GB/s after pinning).
2. **Prefetchers OFF** (`wrmsr -a 0x1a4 0xf`): demand-miss ≈ actual DRAM, AND
   it removed an n_kv-dependent jaggedness (prefetch effectiveness varied with
   n_kv). With prefetch on the curve zigzags; off it is smooth.

Metric reported as **bytes/cycle** (= GB/s ÷ 2.1) — frequency-invariant, so it
sidesteps the sim's `CPU_HZ=3.2 GHz` assumption vs the real 2.1 GHz base.

## Result (`hw_kv_demand_summary.csv`, steady-state median of tok≥5)
| n_kv | DRAM GB/s @2.1 | bytes/cycle |
|---|---|---|
| 256 | 2.26 | 1.07 |
| 512 | 2.41 | 1.15 |
| 768 | 2.49 | 1.18 |
| 1024 | 2.53 | 1.21 |
| 1536 | 2.54 | 1.21 |
| 2048 | 1.83 | 0.87 |

**`hw_vs_sim_kv_demand.csv`** puts both curves side-by-side (bytes/cycle +
`hw_over_sim` ratio). The sim is **compute-bound** (set by `COMPUTE_PER_KV_LINE=32`)
→ **n_kv-flat at 0.984–0.988 bytes/cycle**; hw is **1.09–1.23× the sim** for
n_kv 256–1536 (sim under-models per-cycle DRAM throughput ~15–20%), then
**0.89×** at 2048 where hw drops below sim (TLB thrash). The
**`sim_bpc_matchedL3`** column re-runs the sim with the LLC set to the host's
27.5 MiB L3 (28160KB, 55-way) instead of 2 MiB: it is **identical to the 2 MiB
column at every n_kv** (Δ<0.3%) — so the LLC byte-definition is NOT a confound,
and the hw/sim gap is the compute model, not the cache (proof in
`sim_llc_size_sweep.csv`). **bytes/cycle is the comparison axis** — the GB/s
columns are at different clocks (hw @2.1 GHz, sim @3.2 GHz) and NOT comparable.

## Findings
- Real single-thread KV-attention demand is **O(2–3) GB/s, compute/MLP-limited,
  NOT DRAM-bound** — refutes the "real ≫3.16" hypothesis. Single-thread f16
  vec_dot caps throughput well below DRAM bandwidth.
- **dram/logical ≈ 1.0** → the 64 MiB KV footprint exceeds the 27.5 MiB L3, so
  real DRAM ≈ the unique KV (rereads mostly L3-absorbed at low n_kv).
- **Real bytes/cycle (1.07–1.21) > sim 0.986** → the sim under-models per-cycle
  DRAM throughput ~15% at the operating point.
- **Shape differs**: sim is n_kv-flat (compute-bound constant); real rises to a
  plateau (768–1536) then drops at 2048 (TLB thrash, 256 MiB KV over 4 KiB pages
  — an effect the sim doesn't model).
- **The gap is the compute model, NOT the LLC**: matching the sim LLC to the real
  27.5 MiB L3 leaves bytes/cycle unchanged (`sim_bpc_matchedL3` ≡ 2 MiB at all
  n_kv). The sim demand is a compute-bound constant (`COMPUTE_PER_KV_LINE=32`);
  the ~15% shortfall and the missing n_kv structure are the compute/timing
  model's, not the cache's. (We do NOT recalibrate 32 to close it — circular.)

## Verdict
Per-core demand **magnitude** (~3 GB/s, right order) is realistic at the
operating point — the study's number is not an artifact. But this is **NOT a
structural validation**: the model is n_kv-flat while real has cache/TLB
structure, and the earlier "3.16 GB/s match" was small-sample noise
(3-tok=3.16 vs 15-tok=2.71) at a curve-crossing, leaning on the 3.2 GHz
assumption. Claim only "demand magnitude realistic," never "model reproduces
real."

## Caveats
- **Single-thread (per-core)** = the "N independent single-core decode streams"
  regime, NOT single-instance multi-thread serving (one real stream uses all
  cores → much higher per-stream demand).
- The sim's `total_read_reqs` (used for its 0.986) sits ~10% above the 64 MiB
  unique footprint (70.6 vs 64). This is **NOT 2 MiB-specific eviction** — it is
  flat from 2 MiB to 32 MiB LLC and collapses to unique only at **≥64 MiB = the
  full footprint** (`sim_llc_size_sweep.csv`). It is the long-reuse tail of the
  GQA 4× reread, absorbed only when the cache holds the whole footprint. So the
  sim's bytes/cycle is **LLC-size-invariant (2–32 MiB)**, and matching the LLC
  to the real 27.5 MiB L3 changes nothing (`sim_bpc_matchedL3` ≡ 2 MiB). (Aside:
  non-pow2 LLC *set counts* mis-index catastrophically — for a non-pow2 total
  size like 27.5 MiB pick ways so size/(64·ways) is pow2, e.g. 28160KB @ 55-way
  → 8192 sets; see `--llc-associativity auto`.)

## Files
- `hw_kv_demand_summary.csv` — hw actual-DRAM curve (n_kv, demand, bytes/cycle).
- `hw_vs_sim_kv_demand.csv` — **hw vs sim side-by-side** (bytes/cycle, both 2 MiB
  and 27.5 MiB-matched sim, `hw_over_sim` ratios, GB/s refs). Sim curve =
  decode-tokens=1 trace at each n_kv, compute=32, n=1, 3.2 GHz (canonical config).
- `sim_llc_size_sweep.csv` — sim demand vs LLC size (2 MiB→128 MiB) on the real
  canonical n_kv=512 trace. Proof that bytes/cycle is LLC-invariant 2–32 MiB and
  cliffs only at ≥64 MiB (full footprint). Raw stats in `sim_raw/`.

### raw/
- `kd_<n_kv>.txt` — **actual-DRAM** (per-thread L3-miss), prefetch-OFF, pinned
  2.1 GHz, per-token (logical + dram). The headline → summary CSV.
- `kp_<n_kv>.txt` — pinned-2.1 logical-bytes demand (bytes/cycle curve);
  `kp_512{,b,c}` are the variance repeats.
- `kvt_<n_kv>.txt` — earlier per-token logical curve (before the L3-miss
  instrumentation / prefetch-off), retained for provenance.

Instrument lives in `llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c` (env-gated, zero
cost when `GGML_KV_TIME` unset). See memory `real-kv-demand-measurement`.
