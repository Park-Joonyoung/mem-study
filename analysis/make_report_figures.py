#!/usr/bin/env python3
"""Generate the figures for report/TECH_REPORT.md from the canonical result
sets (no legacy archives, no tmp probes):

  results/core_sweep_b1/      synthetic core sweep, b=1, n=1..128 (23 pts)
  results/batch_sweep_n8/     synthetic batch sweep, n=8, b=1..31 (8 pts)
  results/compute_sweep/      compute-per-line x cores grid (16 pts)
  results/real_core_sweep/    real llama.cpp trace, n=1..12,16,24,32

The bare-named dirs above hold the reread generation: the report text quotes
their numbers, and the real llama.cpp trace carries GQA re-reads natively, so
real-vs-synthetic validation (fig1/7/8/9) only lines up against reread.

plus, when present, the naive 1:1 mirror (same grids/LLC rule, only
--gqa-emission differs) for the emission-mode comparison figures (fig10/11):

  results/{core_sweep_b1, batch_sweep_n8, compute_sweep}_naive/

Writes PNGs to report/figures/ and prints the derived numbers quoted in the
report (knee interpolation, ceiling, batch flatness, demand-halving ratios)
so text and figures stay consistent.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sim_constants import CPU_HZ, CACHE_LINE, DDR4_PEAK_BW_GBS  # noqa: E402

OUT = REPO / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

C_REAL, C_SYN, C_GRID = "#d62728", "#1f77b4", ["#2ca02c", "#1f77b4", "#ff7f0e", "#9467bd"]


def load(path: str) -> list[dict]:
    with (REPO / path).open() as f:
        return [{k: (v if k == "dram" else float(v)) for k, v in r.items() if v != ""}
                for r in csv.DictReader(f)]


core = sorted(load("results/core_sweep_b1/core_batch_sweep_summary.csv"),
              key=lambda r: r["cores"])
batch = sorted(load("results/batch_sweep_n8/core_batch_sweep_summary.csv"),
               key=lambda r: r["batch"])
comp = sorted(load("results/compute_sweep/compute_sweep_summary.csv"),
              key=lambda r: (r["compute"], r["cores"]))
real = sorted(load("results/real_core_sweep/real_core_sweep_summary.csv"),
              key=lambda r: r["cores"])

col = lambda rows, k: [r[k] for r in rows]
sel = lambda rows, k, v: [r for r in rows if r[k] == v]


def style(ax, xlabel, ylabel, title=None):
    ax.grid(alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=10)


# =============================================================================
# Fig 1: real vs synthetic (n=1..8) — runtime / latency / queue / row_hit
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), constrained_layout=True)
core18 = [r for r in core if r["cores"] <= 8]
real18 = [r for r in real if r["cores"] <= 8]   # match synthetic range for the n=1–8 validation
for ax, key, ylab in zip(
        axes.flat,
        ["runtime_cycles", "avg_read_latency_ns", "avg_read_queue_len", "row_hit_rate"],
        ["runtime (cycles)", "avg read latency (ns)", "avg read queue len", "row hit rate"]):
    ax.plot(col(core18, "cores"), col(core18, key), "o-", color=C_SYN, label="synthetic (reread)")
    ax.plot(col(real18, "cores"), col(real18, key), "s--", color=C_REAL, label="real llama.cpp trace")
    style(ax, "cores (concurrent decode streams)", ylab)
axes[0][0].legend(fontsize=9)
fig.suptitle("Fig 1. Real llama.cpp trace vs synthetic generator, b=1, n=1–8 (DDR4-2400 1ch)")
fig.savefig(OUT / "fig1_real_vs_synthetic.png", dpi=150)
plt.close(fig)

# =============================================================================
# Fig 2: core sweep saturation, n=1..128
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), constrained_layout=True)
panels = [("runtime_cycles", "runtime (cycles)"),
          ("avg_read_latency_ns", "avg read latency (ns)"),
          ("avg_read_queue_len", "avg read queue len"),
          ("row_hit_rate", "row hit rate")]
for ax, (key, ylab) in zip(axes.flat, panels):
    ax.plot(col(core, "cores"), col(core, key), "o-", color=C_SYN)
    style(ax, "cores", ylab)
axes[0][1].axhline(2 * core[0]["avg_read_latency_ns"], ls=":", c="gray")
axes[0][1].text(20, 2 * core[0]["avg_read_latency_ns"] + 3, "2x unloaded", fontsize=8, color="gray")
axes[1][0].axhline(32, ls=":", c="gray")
axes[1][0].text(1.2, 32.5, "queue capacity (32)", fontsize=8, color="gray")
fig.suptitle("Fig 2. Core sweep saturation, b=1, n=1–128")
fig.savefig(OUT / "fig2_core_sweep_saturation.png", dpi=150)
plt.close(fig)

# =============================================================================
# Fig 3: bandwidth vs cores + ceiling
# =============================================================================
ceiling = max(col(core, "read_bw_gbs"))   # peak (n24 = 17.70 GB/s); §F4 uses ceiling(peak)/demand for the knee
n1bw = core[0]["read_bw_gbs"]
fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
ax.plot(col(core, "cores"), col(core, "read_bw_gbs"), "o-", color=C_SYN,
        label="measured read BW")
ax.plot([0, DDR4_PEAK_BW_GBS / n1bw], [0, DDR4_PEAK_BW_GBS], ls="--", c="gray", lw=1,
        label=f"linear demand ({n1bw:.2f} GB/s x n)")
ax.axhline(DDR4_PEAK_BW_GBS, ls=":", c="black")
ax.text(22, DDR4_PEAK_BW_GBS + 0.2, "theoretical peak 19.2", fontsize=8)
ax.axhline(ceiling, ls=":", c=C_REAL)
ax.text(22, ceiling - 0.8, f"measured ceiling {ceiling:.1f}", fontsize=8, color=C_REAL)
knee_bw = ceiling / n1bw
ax.axvline(knee_bw, ls=":", c="green")
ax.text(knee_bw + 0.4, 4, f"knee = ceiling/demand\n= {knee_bw:.1f} cores", fontsize=8, color="green")
ax.set_xlim(0, 33)
style(ax, "cores", "total read bandwidth (GB/s)")
ax.legend(fontsize=8, loc="lower right")
fig.suptitle("Fig 3. Channel supply: demand vs measured ceiling")
fig.savefig(OUT / "fig3_bandwidth_ceiling.png", dpi=150)
plt.close(fig)

# =============================================================================
# Fig 4: batch sweep flatness
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
bs = col(batch, "batch")
ax1.plot(bs, [r["runtime_cycles"] / r["batch"] / 1e6 for r in batch], "o-", color=C_SYN)
ax1.set_ylim(110, 130)
style(ax1, "batch (sequences per core, serial)", "runtime per batch unit (Mcycles)")
b1 = batch[0]
for key, lab in [("per_core_ipc", "IPC"), ("avg_read_latency_ns", "latency"),
                 ("avg_read_queue_len", "queue len"), ("row_hit_rate", "row hit"),
                 ("read_bw_gbs", "read BW")]:
    ax2.plot(bs, [r[key] / b1[key] for r in batch], "o-", label=lab)
ax2.set_ylim(0.9, 1.1)
style(ax2, "batch", "metric normalized to b=1")
ax2.legend(fontsize=8, ncol=2)
fig.suptitle("Fig 4. Batch sweep at n=8 (saturated): batch scales work, not contention")
fig.savefig(OUT / "fig4_batch_flatness.png", dpi=150)
plt.close(fig)

# =============================================================================
# Fig 5: compute sweep — demand law + total BW
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
computes = sorted({r["compute"] for r in comp})
n1rows = sorted(sel(comp, "cores", 1), key=lambda r: r["compute"])
meas = [r["read_bw_gbs"] for r in n1rows]
ax1.plot(computes, meas, "o-", color=C_SYN, label="measured (n=1)")
c32bw = [r for r in n1rows if r["compute"] == 32][0]["read_bw_gbs"]
model = [c32bw * 33 / (c + 1) for c in computes]
ax1.plot(computes, model, "x--", c="gray", label="model  BW ∝ 1/(compute+1)")
ax1.set_xscale("log", base=2)
ax1.set_xticks(computes)
ax1.set_xticklabels([int(c) for c in computes])
style(ax1, "compute insts per line (miss spacing - 1)", "per-core read BW (GB/s)")
ax1.legend(fontsize=8)
for c, colr in zip(computes, C_GRID):
    rows = sorted(sel(comp, "compute", c), key=lambda r: r["cores"])
    ax2.plot(col(rows, "cores"), col(rows, "read_bw_gbs"), "o-", color=colr,
             label=f"c={int(c)}")
ax2.axhline(ceiling, ls=":", c="black")
ax2.text(1.05, ceiling - 1.2, f"ceiling {ceiling:.1f}", fontsize=8)
style(ax2, "cores", "total read BW (GB/s)")
ax2.legend(fontsize=8)
fig.suptitle("Fig 5. compute-per-line: per-core demand law and total bandwidth")
fig.savefig(OUT / "fig5_compute_demand.png", dpi=150)
plt.close(fig)

# =============================================================================
# Fig 6: burstiness — latency vs utilization (compute grid + core sweep c32)
# =============================================================================
fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
for c, colr in zip(computes, C_GRID):
    rows = sorted(sel(comp, "compute", c), key=lambda r: r["cores"])
    ax.plot([r["read_bw_gbs"] / DDR4_PEAK_BW_GBS * 100 for r in rows],
            col(rows, "avg_read_latency_ns"), "o-", color=colr, label=f"c={int(c)} (grid n=1–4)")
ext = [r for r in core if 5 <= r["cores"] <= 12]
ax.plot([r["read_bw_gbs"] / DDR4_PEAK_BW_GBS * 100 for r in ext],
        col(ext, "avg_read_latency_ns"), "s--", color=C_GRID[1], alpha=0.5,
        label="c=32 (core sweep n=5–12)")
ax.axvline(ceiling / DDR4_PEAK_BW_GBS * 100, ls=":", c="black")
ax.text(ceiling / DDR4_PEAK_BW_GBS * 100 - 23, 100, f"measured ceiling ({ceiling:.1f})", fontsize=8)
style(ax, "channel utilization (% of 19.2 GB/s peak)", "avg read latency (ns)")
ax.legend(fontsize=8)
fig.suptitle("Fig 6. Same utilization, different latency: burstiness (c16) inflates queueing")
fig.savefig(OUT / "fig6_burstiness.png", dpi=150)
plt.close(fig)

# =============================================================================
# Fig 7: row_hit microstructure
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
c32 = [r for r in core if r["cores"] <= 32]
ax1.plot(col(c32, "cores"), col(c32, "row_hit_rate"), "o-", color=C_SYN, label="synthetic (n=1..32)")
ax1.plot(col(real, "cores"), col(real, "row_hit_rate"), "s--", color=C_REAL, label="real trace (n=1..8)")
ax1.annotate("n2 dip", xy=(2, sel(core, "cores", 2)[0]["row_hit_rate"]),
             xytext=(2.5, 0.962), fontsize=8, arrowprops=dict(arrowstyle="->"))
style(ax1, "cores", "row hit rate")
ax1.legend(fontsize=8)
for n, mk in [(1, "o-"), (2, "s--"), (4, "^:")]:
    rows = sorted(sel(comp, "cores", n), key=lambda r: r["compute"])
    ax2.plot(col(rows, "compute"), col(rows, "row_hit_rate"), mk, label=f"n={n}")
ax2.set_xscale("log", base=2)
ax2.set_xticks(computes)
ax2.set_xticklabels([int(c) for c in computes])
style(ax2, "compute insts per line", "row hit rate")
ax2.legend(fontsize=8)
fig.suptitle("Fig 7. row_hit fine structure: small-n n2 dip (left), traffic-density dependence (right)")
fig.savefig(OUT / "fig7_rowhit_structure.png", dpi=150)
plt.close(fig)

# =============================================================================
# Fig 8: Intel MLC-style loaded-latency curve (x = achieved BW, y = latency)
# =============================================================================
fig, ax = plt.subplots(figsize=(6.8, 4.4), constrained_layout=True)
ax.plot(col(core, "read_bw_gbs"), col(core, "avg_read_latency_ns"), "o-",
        color=C_SYN, label="synthetic core sweep (n = 1..128)")
ax.plot(col(real, "read_bw_gbs"), col(real, "avg_read_latency_ns"), "s--",
        color=C_REAL, label="real llama.cpp trace")
for n in [1, 4, 8, 12, 16, 24, 32]:
    rows = sel(core, "cores", n)
    if rows:
        r = rows[0]
        ax.annotate(f"n={n}", (r["read_bw_gbs"], r["avg_read_latency_ns"]),
                    textcoords="offset points", xytext=(-4, 7), fontsize=7.5,
                    color=C_SYN)
idle = sel(sel(comp, "compute", 128), "cores", 1)[0]["avg_read_latency_ns"]
ax.axhline(idle, ls=":", c="gray", lw=1)
ax.text(0.4, idle + 2, f"unloaded latency ~{idle:.0f} ns", fontsize=8, color="gray")
ax.axvline(ceiling, ls=":", c=C_REAL, lw=1)
ax.text(ceiling - 0.55, 75, f"measured ceiling {ceiling:.1f} GB/s",
        rotation=90, fontsize=8, color=C_REAL)
ax.axvline(DDR4_PEAK_BW_GBS, ls=":", c="black", lw=1)
ax.text(DDR4_PEAK_BW_GBS - 0.55, 75, "theoretical peak 19.2", rotation=90, fontsize=8)
ax.set_xlim(0, 20)
style(ax, "achieved read bandwidth (GB/s)", "avg read latency (ns)")
ax.legend(fontsize=8, loc="upper left")
fig.suptitle("Fig 8. Loaded-latency curve (Intel MLC convention), DDR4-2400 single channel")
fig.savefig(OUT / "fig8_loaded_latency.png", dpi=150)
plt.close(fig)

# =============================================================================
# Fig 9: Kleinrock power P = BW/latency — parameter-free knee
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.0), constrained_layout=True)
def power(rows):
    return [r["read_bw_gbs"] / r["avg_read_latency_ns"] for r in rows]
p_syn, p_real = power(core), power(real)
ax1.plot(col(core, "cores"), p_syn, "o-", color=C_SYN, label="synthetic")
ax1.plot(col(real, "cores"), p_real, "s--", color=C_REAL, label="real trace")
for rows, ps, colr in [(core, p_syn, C_SYN), (real, p_real, C_REAL)]:
    i = max(range(len(ps)), key=lambda k: ps[k])
    n_pk = rows[i]["cores"]
    ax1.annotate(f"peak n={n_pk:.0f}", (n_pk, ps[i]), textcoords="offset points",
                 xytext=(8, 4), fontsize=8, color=colr)
    ax1.plot([n_pk], [ps[i]], "*", ms=12, color=colr)
ax1.set_xscale("log", base=2)
ax1.set_xticks([1, 2, 4, 8, 16, 32])
ax1.set_xticklabels([1, 2, 4, 8, 16, 32])
style(ax1, "cores", "power = BW / latency  (arb. units)")
ax1.legend(fontsize=8)
# Geometric view: power peak = tangent point of a ray from the origin
ax2.plot(col(core, "read_bw_gbs"), col(core, "avg_read_latency_ns"), "o-",
         color=C_SYN, label="synthetic loaded-latency")
ax2.plot(col(real, "read_bw_gbs"), col(real, "avg_read_latency_ns"), "s--",
         color=C_REAL, label="real loaded-latency")
for rows, ps, colr in [(core, p_syn, C_SYN), (real, p_real, C_REAL)]:
    i = max(range(len(ps)), key=lambda k: ps[k])
    bw_pk, lat_pk = rows[i]["read_bw_gbs"], rows[i]["avg_read_latency_ns"]
    ax2.plot([0, 20], [0, 20 * lat_pk / bw_pk], ":", color=colr, lw=1)
    ax2.plot([bw_pk], [lat_pk], "*", ms=12, color=colr)
ax2.set_xlim(0, 20)
ax2.set_ylim(0, 160)
ax2.text(3.2, 30, "rays from origin:\nmin slope = max P\n= tangent at the knee",
         fontsize=8, color="gray")
style(ax2, "achieved read bandwidth (GB/s)", "avg read latency (ns)")
ax2.legend(fontsize=8, loc="upper left")
fig.suptitle("Fig 9. Kleinrock power P = throughput/latency: peak at n=5 = "
             "origin-tangent of the loaded-latency curve")
fig.savefig(OUT / "fig9_power.png", dpi=150)
plt.close(fig)

# =============================================================================
# Fig 10/11: naive vs reread (1:1 emission-mode comparison mirror)
# =============================================================================
C_AMO = "#7f7f7f"
amo_core_p = REPO / "results/core_sweep_b1_naive/core_batch_sweep_summary.csv"
amo_batch_p = REPO / "results/batch_sweep_n8_naive/core_batch_sweep_summary.csv"
amo_comp_p = REPO / "results/compute_sweep_naive/compute_sweep_summary.csv"
if amo_core_p.exists() and amo_batch_p.exists() and amo_comp_p.exists():
    amo_core = sorted(load(str(amo_core_p.relative_to(REPO))), key=lambda r: r["cores"])
    amo_batch = sorted(load(str(amo_batch_p.relative_to(REPO))), key=lambda r: r["batch"])
    amo_comp = sorted(load(str(amo_comp_p.relative_to(REPO))),
                      key=lambda r: (r["compute"], r["cores"]))

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7), constrained_layout=True)
    a, b, c_ax, d = axes.flat
    # (a) runtime vs cores
    a.plot(col(core, "cores"), col(core, "runtime_cycles"), "o-", color=C_SYN, label="reread")
    a.plot(col(amo_core, "cores"), col(amo_core, "runtime_cycles"), "^-", color=C_AMO, label="naive")
    style(a, "cores", "runtime (cycles)", "(a) runtime scaling")
    a.legend(fontsize=8)
    # (b) loaded-latency
    b.plot(col(core, "read_bw_gbs"), col(core, "avg_read_latency_ns"), "o-", color=C_SYN, label="reread")
    b.plot(col(amo_core, "read_bw_gbs"), col(amo_core, "avg_read_latency_ns"), "^-", color=C_AMO, label="naive")
    b.axvline(ceiling, ls=":", c="black", lw=1)
    style(b, "achieved read BW (GB/s)", "avg read latency (ns)",
          "(b) loaded-latency: same ceiling, different knee")
    b.legend(fontsize=8)
    # (c) Kleinrock power, each curve's argmax = its knee
    pw = lambda rows: [r["read_bw_gbs"] / r["avg_read_latency_ns"] for r in rows]
    for rows, colr, lab in [(core, C_SYN, "reread"), (amo_core, C_AMO, "naive")]:
        p = pw(rows)
        c_ax.plot(col(rows, "cores"), p, "o-" if lab == "reread" else "^-", color=colr, label=lab)
        i = max(range(len(p)), key=lambda k: p[k])
        c_ax.plot([rows[i]["cores"]], [p[i]], "*", ms=13, color=colr)
        c_ax.annotate(f"peak n={rows[i]['cores']:.0f}", (rows[i]["cores"], p[i]),
                      textcoords="offset points", xytext=(6, 4), fontsize=8, color=colr)
    c_ax.set_xscale("log", base=2)
    c_ax.set_xticks([1, 2, 4, 8, 16, 32]); c_ax.set_xticklabels([1, 2, 4, 8, 16, 32])
    style(c_ax, "cores", "power = BW/lat (arb.)", "(c) power: argmax positions (levels not comparable)")
    c_ax.legend(fontsize=8)
    # (d) per-core demand vs compute flag at n=1. Same flag value = same total
    # compute = same AVERAGE insts-per-miss in both modes; the vertical gap is
    # purely the miss-clustering (burstiness) effect of the emission structure.
    for rows, colr, mk, lab in [(comp, C_SYN, "o-", "reread"), (amo_comp, C_AMO, "^-", "naive")]:
        n1r = sorted(sel(rows, "cores", 1), key=lambda r: r["compute"])
        d.plot(col(n1r, "compute"), col(n1r, "read_bw_gbs"), mk, color=colr, label=lab)
    d.set_xscale("log", base=2)
    d.set_xticks(computes); d.set_xticklabels([int(x) for x in computes])
    style(d, "compute per line (same avg insts/miss in both modes)",
          "per-core read BW (GB/s)", "(d) demand gap = pure miss-clustering effect")
    d.legend(fontsize=8)
    fig.suptitle("Fig 10. Emission mode 1:1: naive vs reread (identical grids, LLC rule, layout)")
    fig.savefig(OUT / "fig10_naive_vs_reread.png", dpi=150)
    plt.close(fig)

    fig, (e, f_ax) = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
    e.plot(col(batch, "batch"), [r["runtime_cycles"] / r["batch"] / 1e6 for r in batch],
           "o-", color=C_SYN, label="reread (saturated @ n8)")
    e.plot(col(amo_batch, "batch"), [r["runtime_cycles"] / r["batch"] / 1e6 for r in amo_batch],
           "^-", color=C_AMO, label="naive (unsaturated @ n8)")
    style(e, "batch", "runtime per batch unit (Mcycles)")
    e.legend(fontsize=8)
    for rows, colr, mk in [(batch, C_SYN, "o-"), (amo_batch, C_AMO, "^-")]:
        b1r = rows[0]
        f_ax.plot(col(rows, "batch"),
                  [r["avg_read_latency_ns"] / b1r["avg_read_latency_ns"] for r in rows],
                  mk, color=colr)
    f_ax.set_ylim(0.9, 1.1)
    style(f_ax, "batch", "latency normalized to b=1",
          "batch≠contention holds in BOTH modes/regimes")
    fig.suptitle("Fig 11. Batch sweep, naive vs reread: batch scales work in either regime")
    fig.savefig(OUT / "fig11_batch_modes.png", dpi=150)
    plt.close(fig)
    print("fig10/fig11 (naive mirror) generated")
else:
    print("naive mirror incomplete -> fig10/fig11 skipped")

# =============================================================================
# Derived numbers quoted in the report
# =============================================================================
print(f"ceiling (peak read BW @ n24)       : {ceiling:.2f} GB/s "
      f"({ceiling / DDR4_PEAK_BW_GBS * 100:.0f}% of peak)")
print(f"n1 demand (c=32)                   : {n1bw:.2f} GB/s")
print(f"knee = ceiling / demand            : {knee_bw:.2f} cores")
l5, l6 = sel(core, "cores", 5)[0], sel(core, "cores", 6)[0]
thr = 2 * core[0]["avg_read_latency_ns"]
knee_lat = 5 + (thr - l5["avg_read_latency_ns"]) / (l6["avg_read_latency_ns"] - l5["avg_read_latency_ns"])
print(f"knee (latency-2x interpolation)    : n = {knee_lat:.2f}")
per_b = [r["runtime_cycles"] / r["batch"] for r in batch]
print(f"batch runtime/b spread             : {min(per_b)/1e6:.1f}–{max(per_b)/1e6:.1f} Mcyc "
      f"({(max(per_b)/min(per_b)-1)*100:.2f}%)")
for a, b in [(32, 64), (64, 128)]:
    ra = sel(sel(comp, "cores", 1), "compute", a)[0]["read_bw_gbs"]
    rb = sel(sel(comp, "cores", 1), "compute", b)[0]["read_bw_gbs"]
    print(f"demand halving c{a}->c{b}            : x{ra/rb:.3f} (model x{(b+1)/(a+1):.3f})")
r16 = sel(sel(comp, "cores", 1), "compute", 16)[0]["read_bw_gbs"]
print(f"c16 vs model                       : {r16:.2f} measured vs {c32bw*33/17:.2f} model")
real8 = sel(real, "cores", 8)[0]["runtime_cycles"]
syn8 = sel(core, "cores", 8)[0]["runtime_cycles"]
for n in [1, 2, 4, 8]:
    rr = sel(real, "cores", n)[0]["runtime_cycles"]
    ss = sel(core, "cores", n)[0]["runtime_cycles"]
    print(f"real-vs-synthetic runtime err n{n:<2}  : {(ss/rr-1)*100:+.1f}%")
print(f"figures written to {OUT}")
