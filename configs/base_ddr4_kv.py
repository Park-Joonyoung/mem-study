#!/usr/bin/env python3
"""base_ddr4_kv — Ramulator 2.1 config for the LLM memory study.

KV-cache decode (Llama-3.1-8B-style) on DDR4.

Ramulator 2.1 has no `ramulator2 -f config.yaml` executable; configs are Python.
Run it (prints a 2.0-style .stats block to stdout):

    python3 configs/base_ddr4_kv.py [num_expected_insts] > results/out.stats

The optional positional arg overrides num_expected_insts (the old
`-p Frontend.num_expected_insts=...`). Needs Python 3.10 (the interpreter the
ramulator nanobind module was built for).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "run"))
import ram21

MEMORY = "ddr4"
TRACE = "/home/mem-study/traces/kv_decode_llama31_8b.trace"
NUM_EXPECTED_INSTS = 600000000


def main() -> None:
    insts = int(sys.argv[1]) if len(sys.argv) > 1 else NUM_EXPECTED_INSTS
    ram21.run_and_print(MEMORY, [TRACE], insts)


if __name__ == "__main__":
    main()
