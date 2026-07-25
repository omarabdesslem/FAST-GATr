"""
Per-kernel timing comparison: Python,  SIMD C++ and visualization

Produces two panels:
  1. SIMD speedup per kernel vs channel count (log scale).
  2. End-to-end timing ratio (SIMD/Python) vs channel count with crossover annotated.

Run with:
    python -m tools.plots.plot_per_kernel_comparison \
        --python-pk  Code/Results/timing_results/2026-05-30/python_per_kernel_seconds_results_2026-05-30_23-57-04.csv \
        --simd-pk    Code/Results/timing_results/2026-05-30/simd_c_per_kernel_seconds_results_2026-05-30_23-50-58.csv \
        --python-e2e Results/timing_results/CSVs/final/python_timing_results_2026-05-30_20-11-29.csv \
        --simd-e2e   Results/timing_results/CSVs/final/simd_c_timing_results_2026-05-30_22-54-00.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
TEAM80_DIR = SCRIPT_DIR.parents[3]
PLOTS_DIR = TEAM80_DIR / "Results" / "timing_results" / "Plots"

DEFAULT_PY_PK   = TEAM80_DIR / "Code/Results/timing_results/2026-05-30/python_per_kernel_seconds_results_2026-05-30_23-57-04.csv"
DEFAULT_SI_PK   = TEAM80_DIR / "Code/Results/timing_results/2026-05-30/simd_c_per_kernel_seconds_results_2026-05-30_23-50-58.csv"
DEFAULT_PY_E2E  = TEAM80_DIR / "Results/timing_results/CSVs/final/python_timing_results_2026-05-30_20-11-29.csv"
DEFAULT_SI_E2E  = TEAM80_DIR / "Results/timing_results/CSVs/final/simd_c_timing_results_2026-05-30_22-54-00.csv"
DEFAULT_OUT     = PLOTS_DIR / "per_kernel_python_vs_simd.png"

BACKGROUND  = "#0f1117"
GRID_COLOR  = "#1e2130"
TEXT_COLOR  = "#e0e6f0"
MUTED       = "#7a8ba0"
ACCENT_RED  = "#ff5555"
ACCENT_YELL = "#ffb86c"

KERNEL_COLORS = {
    "equi_linear":       "#ff5555",
    "geometric_product": "#4fc3f7",
    "equi_join":         "#bd93f9",
    "equi_rms_norm":     "#50fa7b",
    "scaler_gated_gelu": "#ffb86c",
}

KERNEL_ORDER = [
    "equi_linear",
    "geometric_product",
    "equi_join",
    "equi_rms_norm",
    "scaler_gated_gelu",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--python-pk",  type=Path, default=DEFAULT_PY_PK)
    p.add_argument("--simd-pk",    type=Path, default=DEFAULT_SI_PK)
    p.add_argument("--python-e2e", type=Path, default=DEFAULT_PY_E2E)
    p.add_argument("--simd-e2e",   type=Path, default=DEFAULT_SI_E2E)
    p.add_argument("--out",        type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def _rcparams() -> None:
    plt.rcParams.update({
        "font.family":      "monospace",
        "axes.facecolor":   BACKGROUND,
        "figure.facecolor": BACKGROUND,
        "text.color":       TEXT_COLOR,
        "axes.labelcolor":  TEXT_COLOR,
        "xtick.color":      MUTED,
        "ytick.color":      MUTED,
        "axes.edgecolor":   GRID_COLOR,
        "grid.color":       GRID_COLOR,
        "grid.linewidth":   0.8,
        "grid.alpha":       1.0,
    })


def load_per_kernel(py_path: Path, si_path: Path) -> pd.DataFrame:
    py = pd.read_csv(py_path)
    si = pd.read_csv(si_path)
    merged = py.merge(
        si, on=["kernel", "batch", "tokens", "channels", "layers"],
        suffixes=("_py", "_si"),
    )
    merged["speedup"] = merged["avg_ms_py"] / merged["avg_ms_si"]
    return merged.sort_values("channels")


def load_e2e(py_path: Path, si_path: Path) -> pd.DataFrame:
    py = pd.read_csv(py_path)
    si = pd.read_csv(si_path)
    merged = py.merge(si, on=["batch", "tokens", "channels", "layers"], suffixes=("_py", "_si"))
    merged["ratio"] = merged["avg_ms_si"] / merged["avg_ms_py"]
    return merged.sort_values("channels")


def plot_speedup(ax: plt.Axes, pk: pd.DataFrame) -> None:
    channels = sorted(pk["channels"].unique())

    # Reference line at y=1.
    ax.axhline(1, color=ACCENT_RED, linewidth=1.2, linestyle="--", alpha=0.6, zorder=2)
    ax.text(channels[-1] * 1.02, 1.05, "breakeven", color=ACCENT_RED, fontsize=8, va="bottom")

    for kname in KERNEL_ORDER:
        sub = pk[pk["kernel"] == kname].sort_values("channels")
        color = KERNEL_COLORS[kname]
        lw = 3.0 if kname == "equi_linear" else 1.8
        zorder = 5 if kname == "equi_linear" else 3
        ax.plot(
            sub["channels"], sub["speedup"],
            color=color, linewidth=lw, marker="o", markersize=6,
            markerfacecolor=BACKGROUND, markeredgecolor=color, markeredgewidth=1.8,
            label=kname, zorder=zorder,
        )

    # Annotate the equi_linear crossover point.
    el = pk[pk["kernel"] == "equi_linear"].sort_values("channels")
    crossover = el[el["speedup"] < 1]
    if not crossover.empty:
        cx = int(crossover.iloc[0]["channels"])
        cy = crossover.iloc[0]["speedup"]
        ax.annotate(
            f"equi_linear flips\nPython faster (C={cx})",
            xy=(cx, cy), xytext=(cx * 1.5, cy * 2.5),
            color=ACCENT_RED, fontsize=8,
            arrowprops=dict(arrowstyle="->", color=ACCENT_RED, lw=1.2),
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Channels (C)", fontsize=11, labelpad=8)
    ax.set_ylabel("SIMD speedup over Python  (>1 = SIMD faster)", fontsize=10, labelpad=8)
    ax.set_title("Per-kernel speedup: SIMD C++ vs Python", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(channels)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.grid(axis="both", linestyle="--", alpha=0.5)
    ax.legend(
        frameon=True, facecolor="#1a1f2e", edgecolor=GRID_COLOR,
        labelcolor=TEXT_COLOR, fontsize=8.5, loc="upper right",
    )


def plot_e2e(ax: plt.Axes, e2e: pd.DataFrame) -> None:
    channels = e2e["channels"].tolist()
    ratios   = e2e["ratio"].tolist()

    # ratio = SIMD/Python: >1 means SIMD is slower (Python wins), <1 means SIMD is faster.
    color_simd_faster   = "#4fc3f7"  # ratio < 1
    color_python_faster = ACCENT_RED  # ratio > 1

    ax.fill_between(channels, ratios, 1,
                    where=[r < 1 for r in ratios], interpolate=True,
                    color=color_simd_faster, alpha=0.14)
    ax.fill_between(channels, ratios, 1,
                    where=[r > 1 for r in ratios], interpolate=True,
                    color=color_python_faster, alpha=0.18)

    ax.axhline(1, color=ACCENT_YELL, linewidth=1.2, linestyle="--", alpha=0.7, zorder=3)

    ax.plot(channels, ratios, color=TEXT_COLOR, linewidth=2.5,
            marker="o", markersize=7,
            markerfacecolor=BACKGROUND, markeredgecolor=TEXT_COLOR,
            markeredgewidth=2, zorder=4)

    # Annotate each point with who wins and by how much.
    for c, r in zip(channels, ratios):
        if r < 1:
            label = f"SIMD {1/r:.1f}x"
            color = color_simd_faster
            offset = -16
        else:
            label = f"Py {r:.1f}x"
            color = color_python_faster
            offset = 10
        ax.annotate(label, xy=(c, r), xytext=(0, offset),
                    textcoords="offset points", ha="center",
                    fontsize=7.5, color=color)

    # Find and mark the crossover (ratio crosses 1 from below as C increases).
    for i in range(len(ratios) - 1):
        if ratios[i] <= 1 and ratios[i+1] > 1:
            ax.axvspan(channels[i], channels[i+1], color=ACCENT_YELL, alpha=0.08)
            mid = (channels[i] + channels[i+1]) / 2
            ax.text(mid, max(ratios) * 0.92, "crossover\n(equi_linear flip)",
                    color=ACCENT_YELL, fontsize=8, ha="center", fontweight="bold")
            break

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Channels (C)", fontsize=11, labelpad=8)
    ax.set_ylabel("End-to-end time ratio  (SIMD time / Python time)", fontsize=10, labelpad=8)
    ax.set_title("End-to-end timing ratio: SIMD C++ / Python\n"
                 "(>1 = SIMD slower, <1 = SIMD faster)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(channels)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.grid(axis="both", linestyle="--", alpha=0.5)

    ax.text(0.03, 0.08, "← SIMD faster",
            transform=ax.transAxes, color=color_simd_faster, fontsize=8.5, alpha=0.9)
    ax.text(0.03, 0.92, "← Python faster (SIMD regression)",
            transform=ax.transAxes, color=color_python_faster, fontsize=8.5, alpha=0.9)


def main() -> None:
    args = parse_args()
    _rcparams()

    pk  = load_per_kernel(args.python_pk,  args.simd_pk)
    e2e = load_e2e(args.python_e2e, args.simd_e2e)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 7))
    fig.patch.set_facecolor(BACKGROUND)

    plot_speedup(ax1, pk)
    plot_e2e(ax2, e2e)

    fig.suptitle(
        "SIMD C++ vs Python  —  per-kernel & end-to-end analysis\n"
        "equi_linear scaling (O(C²) + cache thrash) drives the end-to-end regression",
        fontsize=13, fontweight="bold", y=1.01, color=TEXT_COLOR,
    )

    for ax in (ax1, ax2):
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COLOR)

    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=180, bbox_inches="tight", facecolor=BACKGROUND)
    print(f"Saved -> {args.out}")
    plt.show()


if __name__ == "__main__":
    main()
