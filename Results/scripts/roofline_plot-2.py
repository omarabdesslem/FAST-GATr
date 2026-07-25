"""
roofline_plot.py
================
Roofline plot for the Optimized-C++ GATr kernels.
All kernels are placed analytically on the roofline using FLOPs and bytes
from flop_byte_counter.py. Measured performance (GFLOP/s) is used for the y-value of each point, so

Usage liek this:
-----
    python roofline_plot.py
    python roofline_plot.py --csv my_timings.csv
    python roofline_plot.py --kernel equi_linear
    python roofline_plot.py --out my_roofline.png --dpi 300
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, str(Path(__file__).parent))
from Results.scripts.flop_byte_counter import count_kernel


# hardware config
HARDWARE = {
    # Peak compute in GFLOP/s
    "peak_gflops":      10000.0,

    "peak_bandwidth_gbs": 1000.0,

    # Label shown on the plot
    "label": "Apple Siicon M2",
}


# color and marker styles
KERNEL_STYLES: dict[str, dict] = {
    "full_forward":       {"color": "#ffffff", "marker": "H"},
}


#style constants
BG      = "#0f1117"
GRID    = "#1e2130"
TEXT    = "#e0e6f0"
MUTED   = "#7a8ba0"
ROOF_CLR = "#ef5350"
L3_CLR   = "#ff8f00"


def _roofline_curve(ai: np.ndarray, peak_gflops: float, bw_gbs: float) -> np.ndarray:
    return np.minimum(peak_gflops, bw_gbs * ai)


def _ridge(peak_gflops: float, bw_gbs: float) -> float:
    return peak_gflops / bw_gbs


def main() -> None:
    parser = argparse.ArgumentParser(description="Roofline plot for GATr kernels")
    parser.add_argument("--csv", default="optimized_c_timing_results_2026-05-21_21-47-33.csv")
    parser.add_argument("--kernel", default=None, help="Plot only this kernel")
    parser.add_argument(
        "--layers",
        type=int,
        default=None,
        help="Number of layers per model (used since CSV lacks a 'layers' column)",
    )
    parser.add_argument(
        "--base-channels",
        type=int,
        default=None,
        help="Base channels used to scale layers (requires --base-layers)",
    )
    parser.add_argument(
        "--base-layers",
        type=int,
        default=None,
        help="Base layers used to scale layers (requires --base-channels)",
    )
    parser.add_argument("--out", default="roofline.png")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    # Load CSV
    df = pd.read_csv(args.csv)
    if not {"batch", "tokens", "channels", "avg_ms"}.issubset(df.columns):
        sys.exit(f"CSV must contain 'batch', 'tokens', 'channels', 'avg_ms' columns (found: {list(df.columns)})")

    has_layers_col = "layers" in df.columns
    if not has_layers_col:
        if args.layers is None and (args.base_channels is None or args.base_layers is None):
            sys.exit(
                "CSV lacks 'layers' column. Provide --layers or (--base-channels and --base-layers)."
            )

    # Build points
    points: list[dict] = []
    for _, row in df.iterrows():
        B = int(row["batch"])
        T = int(row["tokens"])
        C_in = int(row["channels"])
        if has_layers_col:
            num_layers = int(row["layers"])
        elif args.layers is not None:
            num_layers = int(args.layers)
        else:
            num_layers = int(round(C_in * args.base_layers / args.base_channels))
        avg_ms = float(row["avg_ms"])

        # Count operations for the whole model (full_forward)
        stats = count_kernel("full_forward", B=B, T=T, C_in=C_in, num_layers=num_layers)
        
        # Calculate real performance: Total GFLOPs / Time in seconds
        total_gflops = stats["flops"] / 1e9
        time_s = avg_ms / 1000.0
        measured_gflops_per_sec = total_gflops / time_s

        points.append({
            "kernel": "full_forward",
            "B": B, "T": T, "C": C_in,
            "ai": stats["intensity"],
            "flops": stats["flops"],
            "bytes": stats["bytes"],
            "perf_y": measured_gflops_per_sec,
            "avg_ms": avg_ms
        })

    if not points:
        sys.exit("No data points generated. Check --kernel name and CSV contents.")

    # Figure setup
    plt.rcParams.update({
        "font.family":       "monospace",
        "axes.facecolor":    BG,
        "figure.facecolor":  BG,
        "text.color":        TEXT,
        "axes.labelcolor":   TEXT,
        "xtick.color":       MUTED,
        "ytick.color":       MUTED,
        "axes.edgecolor":    GRID,
        "grid.color":        GRID,
        "grid.linewidth":    0.8,
    })

    fig, ax = plt.subplots(figsize=(12, 7))

    # Roofline curves
    all_ai  = [p["ai"] for p in points]
    ai_min  = max(1e-3, min(all_ai) * 0.1)
    ai_max  = max(all_ai) * 10
    ai_x    = np.logspace(math.log10(ai_min), math.log10(ai_max), 400)
    hw      = HARDWARE

    # DRAM roof
    dram_roof = _roofline_curve(ai_x, hw["peak_gflops"], hw["peak_bandwidth_gbs"])
    ax.plot(ai_x, dram_roof, color=ROOF_CLR, linewidth=2.0,
            label=f"DRAM roof  ({hw['peak_bandwidth_gbs']:.0f} GB/s | {hw['peak_gflops']:.0f} GFLOP/s)")

    ridge_dram = _ridge(hw["peak_gflops"], hw["peak_bandwidth_gbs"])
    ax.axvline(ridge_dram, color=ROOF_CLR, linewidth=0.8, linestyle=":", alpha=0.6)
    ax.text(ridge_dram * 1.05, hw["peak_gflops"] * 0.55,
            f"ridge\n{ridge_dram:.1f} F/B",
            color=ROOF_CLR, fontsize=7.5, va="center")

    # Peak compute ceiling
    ax.axhline(hw["peak_gflops"], color=ROOF_CLR, linewidth=1.0, linestyle="--", alpha=0.5)
    ax.text(ai_min * 1.05, hw["peak_gflops"] * 1.03,
            f"Peak {hw['peak_gflops']:.0f} GFLOP/s",
            color=ROOF_CLR, fontsize=8)

    # Kernel scatter (all placed analytically on the DRAM roofline)
    plotted_kernels: set[str] = set()

    for p in points:
        kname  = p["kernel"]
        sty    = KERNEL_STYLES.get(kname, {"color": "#aaaaaa", "marker": "o"})
        ai     = p["ai"]
        perf_y = p["perf_y"]  # Use actual measured performance!

        label = kname if kname not in plotted_kernels else None
        ax.scatter(
            ai, perf_y,
            color=sty["color"], marker=sty["marker"],
            s=100, alpha=0.90, edgecolors="white", linewidths=0.5,
            label=label, zorder=4,
        )
        ax.annotate(
            f"T={p['T']} C={p['C']}",
            xy=(ai, perf_y), xytext=(6, 4),
            textcoords="offset points",
            fontsize=6.5, color=sty["color"], alpha=0.85,
        )
        plotted_kernels.add(kname)

    # Axes and styling
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity  [FLOPs / Byte]", fontsize=12, labelpad=10)
    ax.set_ylabel("Performance  [GFLOP/s]",               fontsize=12, labelpad=10)
    ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext())
    ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext())
    ax.set_xlim(ai_min, ai_max)
    ax.set_ylim(0.01, hw["peak_gflops"] * 3)
    ax.grid(which="both", linestyle="--", alpha=0.35)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)

    ax.set_title(
        f"Roofline — Optimized C++ (analytical)\n({hw['label']})",
        fontsize=14, fontweight="bold", color=TEXT, pad=14,
    )

    ax.legend(
        frameon=True, facecolor="#1a1f2e", edgecolor=GRID,
        labelcolor=TEXT, fontsize=7.5, loc="lower right",
        ncol=2, columnspacing=0.8, handletextpad=0.4,
    )

    # Save and print summary
    plt.tight_layout()
    plt.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved → {args.out}\n")

    print(f"{'Kernel':<22} {'T':>6} {'C':>5} {'GFLOPs':>9} {'MB':>8} {'AI (F/B)':>10} {'Measured GFLOP/s':>18}")
    print("-" * 84)
    for p in points:
        print(f"{p['kernel']:<22} {p['T']:>6} {p['C']:>5} "
              f"{p['flops']/1e9:>9.4f} {p['bytes']/1e6:>8.2f} "
              f"{p['ai']:>10.4f} {p['perf_y']:>18.2f}")

    plt.show()


if __name__ == "__main__":
    main()
