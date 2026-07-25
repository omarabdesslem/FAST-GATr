"""
Per-kernel timing line plots: baseline C++, optimized C++, SIMD C++, Python.

One subplot per kernel. Each line is one implementation; x = channel count.

Run with:
    python -m tools.plots.plot_per_kernel_all_impls

Or override any CSV:
    python -m tools.plots.plot_per_kernel_all_impls \
        --simd  Code/Results/timing_results/2026-05-31/simd_c_per_kernel_seconds_results_2026-05-31_00-29-42.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
TEAM80_DIR = SCRIPT_DIR.parents[3]
RES_DIR    = TEAM80_DIR / "Code" / "Results" / "timing_results"
PLOTS_DIR  = TEAM80_DIR / "Results" / "timing_results" / "Plots"

# Most recent file per implementation — override with --baseline / --optimized / --simd / --python
DEFAULTS = {
    "baseline":  RES_DIR / "2026-05-30" / "baseline_c_per_kernel_seconds_results_2026-05-30_23-39-07.csv",
    "optimized": RES_DIR / "2026-05-30" / "optimized_c_per_kernel_seconds_results_2026-05-30_23-19-55.csv",
    "simd":      RES_DIR / "2026-05-30" / "simd_c_per_kernel_seconds_results_2026-05-30_23-50-58.csv",
    "python":    RES_DIR / "2026-05-30" / "python_per_kernel_seconds_results_2026-05-30_23-57-04.csv",
}

IMPL_STYLE = {
    "C++":            {"color": "#4fc3f7", "marker": "o",  "lw": 1.8, "label": "Baseline C++"},
    "Optimized C++":  {"color": "#50fa7b", "marker": "s",  "lw": 1.8, "label": "Optimized C++"},
    "Vectorized C++": {"color": "#bd93f9", "marker": "^",  "lw": 2.2, "label": "SIMD C++"},
    "Python":         {"color": "#ffb86c", "marker": "D",  "lw": 1.8, "label": "Python"},
}

KERNEL_ORDER = [
    "geometric_product",
    "equi_linear",
    "equi_join",
    "equi_rms_norm",
    "scaler_gated_gelu",
]

BACKGROUND = "#0f1117"
GRID_COLOR = "#1e2130"
TEXT_COLOR = "#e0e6f0"
MUTED      = "#7a8ba0"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline",  type=Path, default=DEFAULTS["baseline"])
    p.add_argument("--optimized", type=Path, default=DEFAULTS["optimized"])
    p.add_argument("--simd",      type=Path, default=DEFAULTS["simd"])
    p.add_argument("--python",    type=Path, default=DEFAULTS["python"])
    p.add_argument("--out",       type=Path, default=PLOTS_DIR / "per_kernel_all_impls.png")
    p.add_argument("--logy", action="store_true", default=True,
                   help="Use log scale on y axis (default: True).")
    p.add_argument("--no-logy", dest="logy", action="store_false")
    return p.parse_args()


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.sort_values("channels")


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


def main() -> None:
    args = parse_args()
    _rcparams()

    sources = {
        "baseline":  args.baseline,
        "optimized": args.optimized,
        "simd":      args.simd,
        "python":    args.python,
    }

    # Load and tag each CSV; skip missing files with a warning
    frames: list[pd.DataFrame] = []
    for key, path in sources.items():
        if not path.exists():
            print(f"[WARN] {key} file not found, skipping: {path}")
            continue
        df = load(path)
        frames.append(df)

    if not frames:
        raise RuntimeError("No CSV files found.")

    all_data = pd.concat(frames, ignore_index=True)

    # Determine which implementation names are present
    impl_names = [n for n in IMPL_STYLE if n in all_data["implementation"].unique()]

    kernels   = [k for k in KERNEL_ORDER if k in all_data["kernel"].unique()]
    n_kernels = len(kernels)
    n_cols    = 3
    n_rows    = (n_kernels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))
    axes = axes.flatten() if n_kernels > 1 else [axes]

    for ax_idx, kname in enumerate(kernels):
        ax  = axes[ax_idx]
        sub = all_data[all_data["kernel"] == kname]

        channels_all = sorted(sub["channels"].unique())

        for impl in impl_names:
            idf   = sub[sub["implementation"] == impl].sort_values("channels")
            if idf.empty:
                continue
            style = IMPL_STYLE[impl]
            chans = idf["channels"].tolist()
            avgs  = idf["avg_ms"].tolist()
            mins  = idf["min_ms"].tolist()
            maxs  = idf["max_ms"].tolist()

            ax.fill_between(chans, mins, maxs, color=style["color"], alpha=0.10)
            ax.plot(
                chans, avgs,
                color=style["color"], linewidth=style["lw"],
                marker=style["marker"], markersize=6,
                markerfacecolor=BACKGROUND, markeredgecolor=style["color"],
                markeredgewidth=1.8, label=style["label"], zorder=3,
            )

        ax.set_title(kname, fontsize=12, fontweight="bold", pad=10, color=TEXT_COLOR)
        ax.set_xlabel("Channels (C)", fontsize=10, labelpad=6)
        ax.set_ylabel("avg time (ms)", fontsize=10, labelpad=6)
        ax.set_xscale("log", base=2)
        if args.logy:
            ax.set_yscale("log")
        ax.set_xticks(channels_all)
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.grid(axis="both", linestyle="--", alpha=0.45)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COLOR)

        if ax_idx == 0:
            ax.legend(
                frameon=True, facecolor="#1a1f2e", edgecolor=GRID_COLOR,
                labelcolor=TEXT_COLOR, fontsize=8.5, loc="upper left",
            )

    # Hide any unused subplots.
    for ax in axes[n_kernels:]:
        ax.set_visible(False)

    fig.suptitle(
        "Per-kernel timing (ms) — Baseline, Optimized, SIMD C++, Python",
        fontsize=14, fontweight="bold", color=TEXT_COLOR, y=1.01,
    )
    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=180, bbox_inches="tight", facecolor=BACKGROUND)
    print(f"Saved -> {args.out}")
    plt.show()


if __name__ == "__main__":
    main()
