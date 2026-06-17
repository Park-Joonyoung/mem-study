#!/usr/bin/env python3
"""Ramulator 2.1 runner shim for the LLM memory study.

Ramulator 2.1 dropped the standalone ``ramulator2 -f config.yaml`` executable.
Simulations are now driven from Python: you build component objects
(``ramulator.frontend.SimpleO3``, ``ramulator.controller.GenericDDR``,
``ramulator.memory_system.GenericDRAM``, ...), construct a
``ramulator.Simulation(frontend, memory_system)``, call ``.run()``, and read
``sim.stats`` (a nested dict).

This module recreates the project's old 2.0 base configs as 2.1 component
factories and, crucially, *flattens* the 2.1 nested stats dict back into the
flat ``key: value`` / ``key_<channel>: value`` text format that 2.0 emitted.
That lets every downstream parser (run_*.py, analysis/compare_stats.py) keep
working unchanged — they still read a ``.stats`` text file.

Key 2.0 -> 2.1 changes baked in here:
  * Invocation: subprocess(ramulator2 -f cfg.yaml -p k=v) -> in-process Python API.
  * Translation: ``RandomTranslation`` was removed -> ``NoTranslation`` (identity).
  * Multi-channel: ``DRAM.org.channel: N`` -> N controllers at the system level.
  * Controller impl: ``Generic`` -> ``GenericDDR`` (DDR4).
  * RowPolicy ``OpenRowPolicy`` -> ``Open``; RefreshManager needs a ``scope``
    (``Rank`` for DDR4).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ramulator 2.1's Python package + its 3.10-built nanobind extension.
# Default: the vendored submodule (ramulator2/, built in place). Override with
# RAMULATOR_PYTHON to point at an external Ramulator build's python/ dir.
import os  # noqa: E402
RAMULATOR_PYTHON = os.environ.get(
    "RAMULATOR_PYTHON",
    str(Path(__file__).resolve().parent.parent / "ramulator2" / "python"),
)
if RAMULATOR_PYTHON not in sys.path:
    sys.path.insert(0, RAMULATOR_PYTHON)

import ramulator  # noqa: E402

# Single-source simulation constants (coupling notes live in sim_constants.py).
# CACHE_LINE / MAX_ADDR are re-exported here for existing importers.
REPO_ROOT = str(Path(__file__).resolve().parents[1])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from sim_constants import (  # noqa: E402,F401
    CACHE_LINE, MAX_ADDR,
    CPU_CLOCK_RATIO,
    DDR4_CHANNELS, DDR4_MEM_CLOCK_RATIO, DDR4_ORG_PRESET, DDR4_RANKS,
    DDR4_TIME_UNIT_NS, DDR4_TIMING_PRESET,
)

# Address mappers, keyed by the names the old YAML/CLI used. Casing matches the
# 2.1 component class names (MOP4CLXOR stays all-caps).
ADDR_MAPPERS = {
    "RoBaRaCoCh": ramulator.addr_mapper.RoBaRaCoCh,
    "ChRaBaRoCo": ramulator.addr_mapper.ChRaBaRoCo,
    "MOP4CLXOR": ramulator.addr_mapper.MOP4CLXOR,
}


def _addr_mapper(name: str):
    try:
        return ADDR_MAPPERS[name]()
    except KeyError:
        raise ValueError(
            f"unknown address mapper {name!r}; known: {sorted(ADDR_MAPPERS)}"
        ) from None


def _ddr4_controller(addr_mapper: str):
    """One DDR4 channel: DDR4_8Gb_x8 @ DDR4_2400R, 2 ranks (was org.rank: 2)."""
    dram = ramulator.dram.DDR4(
        org_preset=DDR4_ORG_PRESET, timing_preset=DDR4_TIMING_PRESET,
        rank=DDR4_RANKS,
    )
    return ramulator.controller.GenericDDR(
        dram=dram,
        scheduler=ramulator.scheduler.FRFCFS(),
        refresh_manager=ramulator.refresh_manager.AllBank(scope="Rank"),
        row_policy=ramulator.row_policy.Open(),
        addr_mapper=_addr_mapper(addr_mapper),
    )


# memory name -> spec. Mirrors the old configs/base_ddr4*.yaml DRAM
# sections, plus the per-standard memory clock_ratio.
#
# clock_ratio is proportional to the simulation *tick* frequency (Ramulator
# ticks memory_system once per `clock_ratio` units; higher = faster). The
# frontend is fixed at clock_ratio=8 ≈ 3.2 GHz CPU. The memory clock_ratio must
# match each standard's tick frequency relative to that CPU:
#
#   DDR4-2400R: tick = CK = tCK 833 ps  -> 1.20 GHz -> clock_ratio 3   (8 × 1.2/3.2)
# time_unit_ns = real time per memory simulation tick (= tCK_ps / 1000), used to
# convert memory-cycle metrics (latency, sim cycles) to ns:
#   DDR4-2400R: 833 ps / 1            = 0.833 ns
MEMORIES = {
    "ddr4": {"channels": DDR4_CHANNELS, "controller": _ddr4_controller,
             "mem_clock_ratio": DDR4_MEM_CLOCK_RATIO,
             "time_unit_ns": DDR4_TIME_UNIT_NS},
}


def build_sim(
    memory: str,
    traces,
    num_expected_insts: int,
    addr_mapper: str = "RoBaRaCoCh",
    channels: int | None = None,
    llc_associativity: int | None = None,
    llc_capacity: str | None = None,
):
    """Build a ramulator.Simulation equivalent to the old base config.

    memory:   "ddr4"
    traces:   iterable of trace file paths (one SimpleO3 core per trace)
    addr_mapper: one of ADDR_MAPPERS (overrides MemorySystem.AddrMapper.impl)
    channels: override the per-memory default channel count
    llc_associativity: override LLC ways (default 8). CAUTION: SimpleO3's LLC
        (2MB x n_cores total) mis-indexes when its set count is not a power of
        two (llc.cpp masks with set_size-1 and floor-log2 tag offset), silently
        turning distinct lines into false hits. Set count = n_cores * 2MB /
        (64B * ways); pass ways = n_cores to pin it at 32768 (pow2) for any
        core count.
    """
    if memory not in MEMORIES:
        raise ValueError(f"unknown memory {memory!r}; known: {sorted(MEMORIES)}")
    spec = MEMORIES[memory]
    n_channels = spec["channels"] if channels is None else channels

    frontend_kwargs = dict(
        clock_ratio=CPU_CLOCK_RATIO,
        traces=[str(t) for t in traces],
        num_expected_insts=int(num_expected_insts),
        translation=ramulator.translation.NoTranslation(max_addr=MAX_ADDR),
    )
    if llc_associativity is not None:
        frontend_kwargs["llc_associativity"] = int(llc_associativity)
    if llc_capacity is not None:
        # Override per-core LLC size (default "2MB"). NOTE: SimpleO3's
        # parse_capacity_str uses stoull -> reads only the leading integer, so
        # "27.5MB" silently becomes 27 MiB; use KB for fractional MiB
        # ("28160KB" == 27.5 MiB). Capacity must also yield a power-of-two set
        # count (size / (64B * ways)) or the index mask mis-indexes (see the
        # llc_associativity caution) -> pick ways so size/(64*ways) is pow2.
        frontend_kwargs["llc_capacity_per_core"] = str(llc_capacity)
    frontend = ramulator.frontend.SimpleO3(**frontend_kwargs)
    memory_system = ramulator.memory_system.GenericDRAM(
        clock_ratio=spec["mem_clock_ratio"],
        controllers=[spec["controller"](addr_mapper) for _ in range(n_channels)],
        channel_mapper=ramulator.channel_mapper.CacheLineInterleave(),
    )
    return ramulator.Simulation(frontend, memory_system)


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def flatten_stats(stats: dict) -> dict:
    """Flatten the 2.1 nested stats dict into 2.0-style flat keys.

    Frontend scalars (incl. per-core ``cycles_recorded_core_<n>``) are kept
    verbatim. Each controller in the ``controller`` list is a DRAM channel; its
    scalar stats are emitted as ``<key>_<channel_index>``. Channel-internal
    per-core counters (``..._core_<n>``) are dropped — downstream code only
    consumes the channel-level aggregates. ``memory_system_cycles`` is
    synthesized from the (equal) per-controller ``cycles``.
    """
    import re

    flat: dict = {}

    frontend = stats.get("frontend", {})
    for key, value in frontend.items():
        if _is_number(value):
            flat[key] = value

    mem = stats.get("memory_system", {})
    for key in (
        "total_num_read_requests",
        "total_num_write_requests",
        "total_num_other_requests",
    ):
        if _is_number(mem.get(key)):
            flat[key] = mem[key]

    controllers = mem.get("controller", [])
    if isinstance(controllers, dict):  # single-channel comes back as one dict
        controllers = [controllers]

    core_suffix = re.compile(r"_core_\d+$")
    max_cycles = 0
    for idx, ctrl in enumerate(controllers):
        if not isinstance(ctrl, dict):
            continue
        for key, value in ctrl.items():
            if not _is_number(value):
                continue
            if core_suffix.search(key):
                continue
            flat[f"{key}_{idx}"] = value
        cyc = ctrl.get("cycles")
        if _is_number(cyc):
            max_cycles = max(max_cycles, cyc)
    flat["memory_system_cycles"] = max_cycles

    return flat


def _format_value(value) -> str:
    # Ints stay int-looking (parsers expect ``\d+``); floats keep a decimal so
    # parsers/compare_stats classify them as floats.
    if isinstance(value, float):
        if value != value:  # NaN
            return ".nan"
        text = repr(value)
        if "." not in text and "e" not in text and "E" not in text:
            text += ".0"
        return text
    return str(value)


def render_stats_text(flat: dict) -> str:
    """Render a flat stats dict as a 2.0-style ``.stats`` text block.

    Grouped under Frontend / MemorySystem / Controller headers for readability;
    the downstream regex parsers only rely on ``key: value`` and indentation is
    tolerated, so the grouping is purely cosmetic.
    """
    import re

    frontend_keys, total_keys, channel_keys = [], [], []
    channel_re = re.compile(r"_\d+$")
    for key in flat:
        if key.startswith("total_num_") or key == "memory_system_cycles":
            total_keys.append(key)
        elif channel_re.search(key) and not key.startswith(
            ("cycles_recorded_core_", "memory_access_cycles_recorded_core_")
        ):
            channel_keys.append(key)
        else:
            frontend_keys.append(key)

    lines = ["Frontend:", "  impl: SimpleO3"]
    for key in frontend_keys:
        lines.append(f"  {key}: {_format_value(flat[key])}")
    lines += ["", "MemorySystem:", "  impl: GenericDRAM"]
    for key in total_keys:
        lines.append(f"  {key}: {_format_value(flat[key])}")
    lines += ["", "  Controller:"]
    for key in sorted(channel_keys):
        lines.append(f"    {key}: {_format_value(flat[key])}")
    lines.append("")
    return "\n".join(lines)


def run_to_stats_file(sim, stats_path) -> dict:
    """Run ``sim`` to completion, write a 2.0-style ``.stats`` text file, and
    return the flattened stats dict."""
    sim.run()
    flat = flatten_stats(sim.stats)
    Path(stats_path).write_text(render_stats_text(flat))
    return flat


def run_and_print(memory, traces, num_expected_insts, addr_mapper="RoBaRaCoCh",
                  channels=None):
    """Convenience for the configs/*.py scripts: build, run, print legacy text
    to stdout (so ``python configs/base_ddr4_kv.py > results/x.stats`` works)."""
    sim = build_sim(memory, traces, num_expected_insts, addr_mapper, channels)
    sim.run()
    sys.stdout.write(render_stats_text(flatten_stats(sim.stats)))
