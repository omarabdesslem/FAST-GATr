"""
Joins flops_per_case.csv with a cycles CSV on (batch, tokens, channels)
and emits flops_per_cycle for each (B,T,C). Runs once for C++ and once for Python.

Usage (from team80/):
    python Code/python/tools/plots/flops_per_cycle.py
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
TEAM80_DIR = SCRIPT_DIR.parents[3]

DATE = datetime.now().strftime("%Y-%m-%d")
RESULTS_DIR = TEAM80_DIR / "Results" / "timing_results" / "CSVs" / DATE

FLOPS_FILE = RESULTS_DIR / "flops_per_case.csv"

JOBS = [
    ("c_cycles_results.csv", "flops_per_cycle_y_axis.csv"),
    ("python_cycles_results.csv", "flops_per_cycle_y_axis_python.csv"),
]


def key(row: dict) -> tuple[int, int, int]:
    return int(row["batch"]), int(row["tokens"]), int(row["channels"])


def load(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def emit(flops_rows: list[dict], cycles_path: Path, out_path: Path) -> None:
    cycles_by_key = {key(r): r for r in load(cycles_path)}

    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "batch", "tokens", "channels",
            "flops_total", "weighted_flops", "avg_cycles",
            "flops_per_cycle", "weighted_flops_per_cycle",
        ])
        for r in flops_rows:
            k = key(r)
            if k not in cycles_by_key:
                print(f"  skipping {k}: no cycles row in {cycles_path.name}")
                continue
            cyc = float(cycles_by_key[k]["avg_cycles"])
            ft = float(r["flops_total"])
            wf = float(r["weighted_flops"])
            w.writerow([
                r["batch"], r["tokens"], r["channels"],
                r["flops_total"], r["weighted_flops"], int(cyc),
                f"{ft / cyc:.6f}", f"{wf / cyc:.6f}",
            ])

    print(f"Wrote {out_path}")


def main() -> None:
    flops_rows = load(FLOPS_FILE)
    for cyc_name, out_name in JOBS:
        emit(flops_rows, RESULTS_DIR / cyc_name, RESULTS_DIR / out_name)


if __name__ == "__main__":
    main()
