#!/usr/bin/env python3
"""Shared helpers for synthetic Ramulator2 SimpleO3 traces."""

from __future__ import annotations

import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRACE_DIR = REPO_ROOT / "traces"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# Single-source constants (re-exported here for gen/ scripts); coupling notes
# live in sim_constants.py.
from sim_constants import (  # noqa: E402,F401
    CACHE_LINE, COMPUTE_PER_KV_LINE, COMPUTE_PER_STORE_LINE,
    DEFAULT_BASE_K, DEFAULT_BASE_V,
)

DEFAULT_BASE = DEFAULT_BASE_K   # legacy alias


def parse_size(value: str) -> int:
    match = re.fullmatch(r"\s*(\d+)\s*([kmgt]?i?b?|)\s*", value, re.IGNORECASE)
    if not match:
        raise ValueError(f"invalid size: {value!r}")

    number = int(match.group(1))
    suffix = match.group(2).lower()
    scale = {
        "": 1,
        "b": 1,
        "k": 1024,
        "kb": 1024,
        "kib": 1024,
        "m": 1024**2,
        "mb": 1024**2,
        "mib": 1024**2,
        "g": 1024**3,
        "gb": 1024**3,
        "gib": 1024**3,
        "t": 1024**4,
        "tb": 1024**4,
        "tib": 1024**4,
    }[suffix]
    return number * scale


def align_down(addr: int, align: int = CACHE_LINE) -> int:  # 주소를 cache line에 맞게 align
    return addr & ~(align - 1)


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def require_multiple(name: str, value: int, granularity: int) -> None:
    if value <= 0 or value % granularity != 0:
        raise ValueError(f"{name} must be a positive multiple of {granularity} bytes")


def emit_read(f, compute: int, addr: int) -> int:       # 읽기 연산 trace 생성
    f.write(f"{compute} {align_down(addr)}\n")          
    return compute + 1                                  # retire하는 연산 수 : 비메모리 연산 + 메모리 연산 1개


def emit_write_alloc(f, compute: int, addr: int) -> int:
    line = align_down(addr)
    f.write(f"{compute} {line} {line}\n")               # write-alloc : 쓰기 전에 먼저 읽어오기
    return compute + 1                                  # retire하는 연산 수 : 비메모리 연산 + 메모리 연산 1개


def ensure_trace_dir() -> None:
    TRACE_DIR.mkdir(parents=True, exist_ok=True)


def print_summary(out: Path, trace_lines: int, estimated_insts: int, first_addr: int, span_bytes: int) -> None:
    print(f"wrote {out}")
    print(f"address range: 0x{first_addr:x}..0x{first_addr + span_bytes - 1:x} ({span_bytes / 1024 / 1024:.2f} MiB)")
    print(f"trace lines={trace_lines}")
    print(f"estimated retired instructions for one pass={estimated_insts}")
