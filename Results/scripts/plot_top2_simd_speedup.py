"""
plot_top2_simd_speedup.py

Finds the two kernels with the highest SIMD speedup over the baseline,
then produces one line plot per kernel comparing Baseline C++ vs SIMD C++.

Run from the Results/scripts/ directory:
    python plot_top2_simd_speedup.py
    python plot_top2_simd_speedup.py --baseline path/to/baseline.csv --simd path/to/simd.csv
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

# ── defaults ────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
TEAM80_DIR = SCRIPT_DIR.parents[1]
RES_DIR    = TEAM80_DIR / "Code" / "Results" / "timing_results" / "2026-05-30"

DEFAULT_BASELINE = RES_DIR / "baseline_c_per_kernel_seconds_results_2026-05-30_23-39-07.csv"
DEFAULT_SIMD     = RES_DIR / "simd_c_per_kernel_seconds_results_2026-05-30_23-50-58.csv"

# ── style ────────────────────────────────────────────────────────────────────
BACKGROUND = "#0f1117"
GRID_COLOR = "#1e2130"
TEXT_COLOR = "#e0e6f0"
MUTED      = "#7a8ba0"

COLOR_BASELINE = "#4fc3f7"   # blue
COLOR_SIMD     = "#bd93f9"   # purple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    p.add_argument("--simd",     type=Path, default=DEFAULT_SIMD)
    p.add_argument("--out",      type=Path,
                   default=SCRIPT_DIR.parent / "timing_results" / "Plots" / "top2_simd_speedup.png")
    return p.parse_args()


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["channels"] = pd.to_numeric(df["channels"], errors="coerce")
    df["avg_ms"]   = pd.to_numeric(df["avg_ms"],   errors="coerce")
    # N = channels (since channels = BASE_C * N = 1 * N)
    df["N"] = df["channels"].astype(int)
    return df.sort_values("N")


def compute_speedup(baseline: pd.DataFrame, simd: pd.DataFrame) -> dict[str, float]:
    """Return mean speedup (baseline / simd) per kernel across all N values."""
    speedups: dict[str, float] = {}
    kernels = set(baseline["kernel"].unique()) & set(simd["kernel"].unique())
    for kname in kernels:
        b = baseline[baseline["kernel"] == kname].set_index("N")["avg_ms"]
        s = simd[simd["kernel"] == kname].set_index("N")["avg_ms"]
        common = b.index.intersection(s.index)
        if common.empty:
            continue
        ratio = b.loc[common] / s.loc[common]
        speedups[kname] = float(ratio.mean())
    return speedups


def apply_style() -> None:
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


def plot_kernel(ax: plt.Axes, kname: str, baseline: pd.DataFrame,
                simd: pd.DataFrame, mean_speedup: float, logy: bool) -> None:
    b = baseline[baseline["kernel"] == kname].sort_values("N")
    s = simd[simd["kernel"] == kname].sort_values("N")

    n_vals = sorted(set(b["N"].tolist()) | set(s["N"].tolist()))

    # Shaded min/max bands
    if {"min_ms", "max_ms"}.issubset(b.columns):
        ax.fill_between(b["N"], b["min_ms"], b["max_ms"],
                        color=COLOR_BASELINE, alpha=0.10)
    if {"min_ms", "max_ms"}.issubset(s.columns):
        ax.fill_between(s["N"], s["min_ms"], s["max_ms"],
                        color=COLOR_SIMD, alpha=0.10)

    # Main lines
    ax.plot(b["N"], b["avg_ms"],
            color=COLOR_BASELINE, linewidth=2.2, marker="o", markersize=6,
            markerfacecolor=BACKGROUND, markeredgecolor=COLOR_BASELINE,
            markeredgewidth=1.8, label="Baseline C++", zorder=3)

    ax.plot(s["N"], s["avg_ms"],
            color=COLOR_SIMD, linewidth=2.2, marker="^", markersize=7,
            markerfacecolor=BACKGROUND, markeredgecolor=COLOR_SIMD,
            markeredgewidth=1.8, label="SIMD C++", zorder=3)

    # Per-point speedup annotations above the SIMD line
    b_idx = b.set_index("N")["avg_ms"]
    s_idx = s.set_index("N")["avg_ms"]
    common = b_idx.index.intersection(s_idx.index)
    for n in common:
        ratio = b_idx[n] / s_idx[n]
        ax.annotate(
            f"{ratio:.1f}×",
            xy=(n, s_idx[n]),
            xytext=(0, 10), textcoords="offset points",
            fontsize=7.5, color=COLOR_SIMD, ha="center", alpha=0.90,
        )

    # Axes
    ax.set_title(
        kname.replace("_", " "),
        fontsize=13, fontweight="bold", color=TEXT_COLOR, pad=12,
    )
    ax.set_xlabel("N  (batch=8, tokens=2N, channels=N, layers=N)",
                  fontsize=9, labelpad=8)
    ax.set_ylabel("Time (ms)", fontsize=10, labelpad=8)
    ax.set_xticks(n_vals)
    ax.set_xticklabels([str(v) for v in n_vals], fontsize=8)

    if logy:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda v, _: f"{v:.3g} ms"))
    else:
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda v, _: f"{v:.2f} ms"))

    ax.tick_params(axis="both", length=4)
    ax.grid(axis="y", linestyle="--")
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_COLOR)

    # Mean speedup badge
    ax.text(
        0.99, 0.04,
        f"mean speedup: {mean_speedup:.1f}×",
        transform=ax.transAxes, fontsize=9, color=COLOR_SIMD,
        va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1f2e",
                  edgecolor=GRID_COLOR, alpha=0.9),
    )

    ax.legend(
        frameon=True, facecolor="#1a1f2e", edgecolor=GRID_COLOR,
        labelcolor=TEXT_COLOR, fontsize=9, loc="upper left",
    )


def main() -> None:
    args = parse_args()
    apply_style()

    baseline = load(args.baseline)
    simd     = load(args.simd)

    # Find top-2 kernels by mean speedup
    speedups = compute_speedup(baseline, simd)
    if len(speedups) < 2:
        raise RuntimeError(f"Need at least 2 kernels in common, found: {list(speedups)}")

    top2 = sorted(speedups, key=speedups.get, reverse=True)[:2]
    print("Top-2 kernels by mean SIMD speedup over baseline:")
    for k in top2:
        print(f"  {k:30s}  {speedups[k]:.2f}×")

    # Two side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        "Top-2 SIMD Speedup Kernels — Baseline C++ vs SIMD C++",
        fontsize=14, fontweight="bold", color=TEXT_COLOR, y=1.02,
    )

    for ax, kname in zip(axes, top2):
        plot_kernel(ax, kname, baseline, simd, speedups[kname], logy=True)

    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=180, bbox_inches="tight")
    print(f"Saved → {args.out}")
    plt.show()


if __name__ == "__main__":
    main()
