#!/usr/bin/env python3
"""
Compare Python and optimized C++ timing CSV files.

By default this script plots:
  python:    Results/timing_results/CSVs/2026-05-20/python_timing_results_2026-05-20_15-46-49.csv
  optimized: Results/timing_results/CSVs/2026-05-20/optimized_c_timing_results_2026-05-20_15-37-13.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
TEAM80_DIR = SCRIPT_DIR.parents[3]
TIMING_DIR = TEAM80_DIR / "Results" / "timing_results"
CSV_DIR = TIMING_DIR / "CSVs"
PLOTS_DIR = TIMING_DIR / "Plots"
DEFAULT_PYTHON = CSV_DIR / "2026-05-20" / "python_timing_results_2026-05-20_15-46-49.csv"
DEFAULT_OPTIMIZED = CSV_DIR / "2026-05-20" / "optimized_c_timing_results_2026-05-20_15-37-13.csv"
DEFAULT_OUT = PLOTS_DIR / "python_vs_optimized_timing.png"
JOIN_KEYS = ["batch", "tokens", "channels"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Python vs optimized C++ timing results.")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON, help="Python timing CSV file.")
    parser.add_argument("--optimized", type=Path, default=DEFAULT_OPTIMIZED, help="Optimized C++ CSV file.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output image path.")
    return parser.parse_args()


def load_results(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = set(JOIN_KEYS + ["avg_ms", "min_ms", "max_ms"])
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    df = df.copy()
    df["label"] = label
    for col in ("avg_ms", "min_ms", "max_ms"):
        df[col.replace("_ms", "_s")] = df[col] / 1_000
    return df


def case_label(row: pd.Series) -> str:
    return f"B={int(row.batch)}, T={int(row.tokens)}, C={int(row.channels)}"


def main() -> None:
    args = parse_args()
    python = load_results(args.python, "Python")
    optimized = load_results(args.optimized, "Optimized C++")

    merged = python.merge(
        optimized,
        on=JOIN_KEYS,
        suffixes=("_python", "_optimized"),
        how="inner",
    ).sort_values(JOIN_KEYS)

    if merged.empty:
        raise ValueError("No matching rows found between the two CSV files.")

    merged["case"] = merged.apply(case_label, axis=1)
    merged["ratio"] = merged["avg_ms_python"] / merged["avg_ms_optimized"]

    # Long form makes grouped plotting simple.
    plot_df = pd.concat(
        [
            merged[JOIN_KEYS + ["case", "avg_s_python", "min_s_python", "max_s_python"]]
            .rename(columns={"avg_s_python": "avg_s", "min_s_python": "min_s", "max_s_python": "max_s"})
            .assign(label="Python"),
            merged[JOIN_KEYS + ["case", "avg_s_optimized", "min_s_optimized", "max_s_optimized"]]
            .rename(columns={"avg_s_optimized": "avg_s", "min_s_optimized": "min_s", "max_s_optimized": "max_s"})
            .assign(label="Optimized C++"),
        ],
        ignore_index=True,
    )

    background = "#0f1117"
    grid_color = "#1e2130"
    text_color = "#e0e6f0"
    muted = "#7a8ba0"
    colors = {"Python": "#ffb86c", "Optimized C++": "#4fc3f7"}

    plt.rcParams.update(
        {
            "font.family": "monospace",
            "axes.facecolor": background,
            "figure.facecolor": background,
            "text.color": text_color,
            "axes.labelcolor": text_color,
            "xtick.color": muted,
            "ytick.color": muted,
            "axes.edgecolor": grid_color,
            "grid.color": grid_color,
            "grid.linewidth": 0.8,
            "grid.alpha": 1.0,
        }
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    x_positions = range(len(merged))

    for label, group in plot_df.groupby("label"):
        group = group.set_index("case").loc[merged["case"]].reset_index()
        color = colors[label]
        ax.fill_between(
            x_positions,
            group["min_s"],
            group["max_s"],
            color=color,
            alpha=0.10,
        )
        ax.plot(
            x_positions,
            group["avg_s"],
            color=color,
            linewidth=2.4,
            marker="o",
            markersize=7,
            markerfacecolor=background,
            markeredgecolor=color,
            markeredgewidth=2,
            label=label,
            zorder=3,
        )

    for idx, row in enumerate(merged.itertuples()):
        if row.ratio >= 1.0:
            ratio_text = f"{row.ratio:.2f}x faster"
        else:
            ratio_text = f"{1.0 / row.ratio:.2f}x slower"
        ax.annotate(
            ratio_text,
            xy=(idx, row.avg_s_optimized),
            xytext=(0, -18),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color=colors["Optimized C++"],
        )

    ax.set_title("Python vs Optimized C++ Timing", fontsize=14, fontweight="bold", pad=16)
    ax.set_xlabel("Test case", fontsize=11, labelpad=10)
    ax.set_ylabel("Time (seconds)", fontsize=11, labelpad=10)
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(merged["case"], rotation=20, ha="right", fontsize=8.5)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f s"))
    ax.grid(axis="y", linestyle="--")
    ax.grid(axis="x", linestyle=":", alpha=0.35)

    for spine in ax.spines.values():
        spine.set_edgecolor(grid_color)

    ax.legend(
        frameon=True,
        facecolor="#1a1f2e",
        edgecolor=grid_color,
        labelcolor=text_color,
        fontsize=9,
        loc="upper left",
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=180, bbox_inches="tight")
    print(f"Compared {len(merged)} matching cases")
    print(f"Saved -> {args.out}")
    plt.show()


if __name__ == "__main__":
    main()
