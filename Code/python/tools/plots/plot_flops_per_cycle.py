"""
Plots FLOPs/cycle vs T for the C++ baseline and the Python baseline.

Reads:
    Results/timing_results/CSVs/<DATE>/flops_per_cycle_y_axis.csv         (C++)
    Results/timing_results/CSVs/<DATE>/flops_per_cycle_y_axis_python.csv  (Python)

Writes:
    Results/timing_results/Plots/flops_per_cycle_plot.png

Usage (from team80/):
    python Code/python/tools/plots/plot_flops_per_cycle.py
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
TEAM80_DIR = SCRIPT_DIR.parents[3]

DATE = datetime.now().strftime("%Y-%m-%d")
CSV_DIR = TEAM80_DIR / "Results" / "timing_results" / "CSVs" / DATE
PLOTS_DIR = TEAM80_DIR / "Results" / "timing_results" / "Plots"

C_FILE = CSV_DIR / "flops_per_cycle_y_axis.csv"
PY_FILE = CSV_DIR / "flops_per_cycle_y_axis_python.csv"
OUTPUT_FILE = PLOTS_DIR / "flops_per_cycle_plot.png"
OUTPUT_FILE_LINEAR_Y = PLOTS_DIR / "flops_per_cycle_plot_linear_y.png"


def load(path: Path) -> tuple[list[int], list[float]]:
    ts, ys = [], []
    with path.open() as f:
        for r in csv.DictReader(f):
            ts.append(int(r["tokens"]))
            ys.append(float(r["flops_per_cycle"]))
    return ts, ys


def render(c_t, c_y, py_t, py_y, *, log_y: bool, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        py_t, py_y,
        label="Python baseline",
        color="tab:blue",
        marker="o",
        markersize=8,
        linewidth=2,
    )
    ax.plot(
        c_t, c_y,
        label="C++ baseline",
        color="tab:red",
        marker="s",
        markersize=8,
        linewidth=2,
    )

    ax.set_xscale("log", base=2)
    if log_y:
        ax.set_yscale("log")
    ax.set_xticks(c_t)
    ax.set_xticklabels([f"$2^{{{t.bit_length() - 1}}}$" for t in c_t])

    ax.set_xlabel("Sequence length T  (B=8, C=2)")
    ax.set_ylabel("FLOPs / cycle")
    ax.set_title("C++ vs Python baseline")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    c_t, c_y = load(C_FILE)
    py_t, py_y = load(PY_FILE)

    render(c_t, c_y, py_t, py_y, log_y=True, out_path=OUTPUT_FILE)
    render(c_t, c_y, py_t, py_y, log_y=False, out_path=OUTPUT_FILE_LINEAR_Y)


if __name__ == "__main__":
    main()
