"""Single source of truth for simulation constants shared across gen/ and run/.

Every value here is load-bearing for result correctness, and several are
COUPLED — change one without its partners and simulations keep running but
produce silently wrong or incomparable numbers. Each block documents what
must move together. Import via repo root (gen/trace_utils.py and run/ram21.py
re-export the common ones for backward compatibility).
"""

# =============================================================================
# Transaction / cache-line granularity
# =============================================================================
# 64 B is pinned BY DESIGN across the whole stack:
#   * traces store 64B-aligned line addresses (gen/trace_utils.py emit_*),
#   * SimpleO3 issues one 64B request per memory instruction,
#   * DDR4: 64-bit channel x prefetch 8  -> tx = exactly 64 B.
# Changing it invalidates every existing trace file.
CACHE_LINE = 64

# =============================================================================
# Clock domains (COUPLED BLOCK — see README "clock_ratio")
# =============================================================================
# Ramulator 2.1 ticks components in proportion to clock_ratio. The frontend
# (SimpleO3 cores) is fixed at ratio 8, calibrated against DDR4-2400R's tick
# (tCK = 833 ps at ratio 3  =>  1 ratio unit = 0.4 GHz):
#   CPU_HZ = 8 x 0.4 GHz = 3.2 GHz.
# runtime_cycles / per-core cycles are in THIS domain. If you change
# CPU_CLOCK_RATIO, recompute CPU_HZ and re-baseline everything.
CPU_CLOCK_RATIO = 8
CPU_HZ = 3.2e9

# Per-standard memory tick ratios + real time per memory tick. These derive
# from the timing presets below — preset and ratio/time_unit move TOGETHER:
#   DDR4_2400R : tick = CK = 833 ps      -> 1.2 GHz  -> ratio 3, 0.833 ns/tick
DDR4_MEM_CLOCK_RATIO = 3
DDR4_TIME_UNIT_NS = 0.833

# =============================================================================
# DRAM organization (COUPLED with the ratios above)
# =============================================================================
DDR4_ORG_PRESET = "DDR4_8Gb_x8"
DDR4_TIMING_PRESET = "DDR4_2400R"
DDR4_RANKS = 2
DDR4_CHANNELS = 1          # single-channel: intra-channel contention study

# Theoretical DDR4-2400 single-channel peak: 8 B bus x 2400 MT/s = 19.2 GB/s.
# Reference only (analysis/discussion); not used to drive the simulator.
DDR4_PEAK_BW_GBS = 19.2

# NoTranslation WRAPS addresses modulo max_addr with no error. Must equal the
# real DDR4 capacity (8Gb x8 device, 8 devices/rank, 2 ranks = 16 GiB) or
# disjoint per-core layouts silently alias (run scripts hard-fail above it).
MAX_ADDR = 1 << 34  # 16 GiB

# SimpleO3's LLC default ways when llc_associativity is not overridden
# (ramulator python wrapper default; LLC is 2MB x n_cores, 64B lines).
# COUPLING: set count = n_cores * 2MB / (64B * ways) must be a power of two or
# the LLC mis-indexes (false hits). Methodology: pow2 core counts use this
# default; non-pow2 use ways = n_cores ('auto'), which pins sets at 32768 —
# but at SMALL n that means low associativity and real-texture traces take
# conflict misses (n3 @ 3-way: +3.5% runtime; clean by 6-way). Prefer 2*n for
# small non-pow2 n. Every result CSV records the effective value (llc_assoc).
LLC_DEFAULT_ASSOC = 8


def llc_auto_assoc(n_cores: int, floor: int = 6) -> int:
    """Minimal LLC ways that keep the set count a power of two, >= floor.

    Set count = n_cores * 2MB / (64B * ways) is pow2 iff ways is a multiple of
    n_cores' odd part; doubling that odd part until >= floor avoids the
    low-associativity conflict-miss seam (measured: 3-way +3.5% runtime on
    real-texture traces, saturated clean by 6-way -> floor 6). For pow2
    n_cores this returns 8 = LLC_DEFAULT_ASSOC, so one rule covers every n
    (e.g. n3->6, n5->10, n7->7, n12->6, n24->6, pow2->8).
    """
    odd = n_cores
    while odd % 2 == 0:
        odd //= 2
    a = odd
    while a < floor:
        a *= 2
    return a

# =============================================================================
# Trace-generation conventions
# =============================================================================
DTYPE_BYTES = 2            # fp16; generator default AND sweep region auto-sizing
REGION_ALIGN = 1 << 20     # per-core K/V regions rounded up to 1 MiB
DEFAULT_BASE_K = 0x1000_0000
DEFAULT_BASE_V = 0x5000_0000

# Compute instructions attached per trace line. This is the REAL<->SYNTHETIC
# CONTRACT: the synthetic generator (per q-head pass in reread mode) and the
# real-trace expander (gen/expand_kv_log.py) must use the SAME values, or
# their instruction counts / miss spacing — and therefore runtime_cycles
# comparisons and the contention knee — silently diverge. 32 also fixes the
# miss spacing at 33 insts, which against SimpleO3's 128-deep window sets
# per-core demand ~3.15 GB/s, matching the measured llama.cpp capture
# (see results/compute_sweep for the sensitivity sweep: demand ~ 1/(c+1)).
COMPUTE_PER_KV_LINE = 32
COMPUTE_PER_STORE_LINE = 2
