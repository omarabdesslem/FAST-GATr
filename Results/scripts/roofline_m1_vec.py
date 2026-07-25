#!/usr/bin/env python3
"""
Roofline plot for the Apple M1 (single-core, scalar + SIMD peaks).

For each measured timing config (batch, tokens, channels, layers) we re-run the
weighted-FLOP counter in Code/python/tools/cpp_flop_counter.py over the whole
GATr forward pass with that build's own kernel math, divide the total weighted
FLOPs by the measured avg_ms to get achieved GFLOP/s, and drop the point at its
analytical arithmetic intensity (weighted FLOPs / estimated byte).

Builds: Python (counted same as baseline, project convention), Baseline C++,
Optimized-for-cache C++, SIMD (cache + packed) C++.

Ceilings are a single-core M1 model, all overridable on the CLI:
    scalar compute : --peak-gflops      (default 25.6 GFLOP/s)
    SIMD compute   : scalar * --simd-width  (default 4x -> 102.4 GFLOP/s)
    bandwidth      : --bandwidth-gbps   (default 60 GB/s)

Where 25.6 comes from: M1 Firestorm P-core at ~3.2 GHz, 4 scalar FP FMAs/cycle
x 2 FLOP/FMA x 3.2 GHz ~= 25.6. One P-core can pull a good chunk of the 68 GB/s
unified bandwidth, so 60 GB/s is the sustained single-core number I went with.

    python roofline_m1_vec.py
    python roofline_m1_vec.py --peak-gflops 12.8 --bandwidth-gbps 60 --simd-width 4
    python roofline_m1_vec.py --out my_roofline.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


SCRIPT_DIR = Path(__file__).resolve().parent
TEAM80_DIR = SCRIPT_DIR.parent.parent          # .../team80
RESULTS_DIR = TEAM80_DIR / "Results"
TOOLS_DIR = TEAM80_DIR / "Code" / "python" / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from cpp_flop_counter import (  # noqa: E402
    Build,
    OpWeights,
    WEIGHTS,
    EquiLinearCfg,
    EquiRMSNormCfg,
    GeometricProductCfg,
    EquiJoinCfg,
    ScalerGatedGeluCfg,
    EquiGeometricAttentionCfg,
    equi_linear_count,
    equi_rms_norm_count,
    geometric_product_count,
    equi_join_count,
    scaler_gated_gelu_count,
    equi_geometric_attention_count,
)


# model topology - has to match flop_byte_counter.count_full_forward / ASLMVOnlyGATr
HIDDEN = 32          # embedding channels_out / block working width
HEADS = 4            # attention heads
ATTN_CQK = 32        # qk channels in attention
ATTN_CV = 32         # v channels in attention
FLOAT_BYTES = 4

# one entry per timing CSV + which build's kernel math the counter uses for it.
# plot=False rows still get written to the CSV but stay off the roofline: Python
# runs as vectorized/multithreaded PyTorch BLAS, so its GFLOP/s isn't comparable
# to a single-core scalar ceiling.
TIMING_SOURCES = [
    {
        "label": "Python",
        "build": Build.BASELINE,          # same FLOPs as baseline (project convention)
        "color": "#ffca28",
        "marker": "o",
        "plot": False,
        "csv": RESULTS_DIR / "python_timing_results_2026-05-30_13-27-38.csv",
    },
    {
        "label": "Baseline C++",
        "build": Build.BASELINE,
        "color": "#ef5350",
        "marker": "s",
        "plot": True,
        "csv": RESULTS_DIR / "baseline_c_timing_results_2026-05-29_12-17-11.csv",
    },
    {
        "label": "Optimized-for-cache C++",
        "build": Build.OPTIMIZED,
        "color": "#4fc3f7",
        "marker": "^",
        "plot": True,
        "csv": RESULTS_DIR / "optimized_for_cache_c_timing_results_2026-05-30_15-06-18.csv",
    },
    {
        "label": "SIMD (cache+packed) C++",
        "build": Build.SIMD,
        "color": "#66bb6a",
        "marker": "D",
        "plot": True,
        "csv": RESULTS_DIR / "simd_with_cache_packed_c_timing_results_2026-05-30_14-07-18.csv",
    },
]


# weighted FLOPs for one full forward pass under a given build
def full_model_weighted_flops(
    *, build: Build, batch: int, tokens: int, channels_in: int, num_layers: int
) -> float:
    """Total weighted FLOPs across the whole model for this build."""
    w: OpWeights = WEIGHTS
    B, T = batch, tokens

    def lin(c_in: int, c_out: int) -> float:
        return equi_linear_count(
            EquiLinearCfg(batch=B, tokens=T, channels_in=c_in,
                          channels_out=c_out, bias=True, build=build)
        ).weighted_flops(w)

    def rms(channels: int) -> float:
        return equi_rms_norm_count(
            EquiRMSNormCfg(n=B * T, channels=channels, hasWt=True, mOne=False, build=build)
        ).weighted_flops(w)

    def gp(m: int) -> float:
        return geometric_product_count(GeometricProductCfg(m=m, build=build)).weighted_flops(w)

    def join(m: int) -> float:
        return equi_join_count(
            EquiJoinCfg(m=m, reference=True, x_nnz_frac=1.0,
                        include_kernel_build=False, noCompOpt=True, build=build)
        ).weighted_flops(w)

    def gelu(m: int) -> float:
        return scaler_gated_gelu_count(ScalerGatedGeluCfg(m=m, build=build)).weighted_flops(w)

    def attn() -> float:
        return equi_geometric_attention_count(
            EquiGeometricAttentionCfg(
                batch=B, heads=HEADS, t_query=T, t_key=T,
                channels_qk=ATTN_CQK, channels_v=ATTN_CV,
                kinds=["ipa", "daa"], build=build,
            )
        ).weighted_flops(w)

    # embedding: EquiLinear(C_in -> HIDDEN)
    total = lin(channels_in, HIDDEN)

    # one block, working width HIDDEN=32
    block = 0.0
    # mlp branch
    block += rms(HIDDEN)                       # mlp layer_norm
    block += lin(HIDDEN, 128)                  # proj_bil  32 -> 128
    block += gp(B * T * HIDDEN)                # geometric_product(32)
    block += join(B * T * HIDDEN)              # equi_join(32)
    block += lin(64, HIDDEN)                   # proj_bil_out 64 -> 32
    block += gelu(B * T * HIDDEN)              # scaler_gated_gelu(32)
    block += lin(HIDDEN, HIDDEN)               # proj_out 32 -> 32
    # attention branch
    block += rms(HIDDEN)                       # attn layer_norm
    block += lin(HIDDEN, 384)                  # proj_qkv 32 -> 384
    block += attn()                            # fused attention
    block += lin(128, HIDDEN)                  # attn proj_out 128 -> 32

    total += num_layers * block

    # head: EquiLinear(HIDDEN -> 1)
    total += lin(HIDDEN, 1)
    return total


# rough byte estimate - same simple tensor-traffic model the estimated roofline
# uses. build-independent on purpose so AI reflects the problem, not the impl.
def estimated_bytes(*, batch: int, tokens: int, channels_in: int, num_layers: int) -> float:
    B, T, C = batch, tokens, channels_in
    input_bytes = B * T * C * 16 * FLOAT_BYTES
    hidden_bytes = B * T * HIDDEN * 16 * FLOAT_BYTES
    qkv_bytes = 3 * B * HEADS * T * HIDDEN * 16 * FLOAT_BYTES
    attn_matrix_bytes = B * HEADS * T * T * FLOAT_BYTES
    per_block = hidden_bytes + qkv_bytes + attn_matrix_bytes + hidden_bytes
    # input/output once, block traffic x layers, then x2 for intermediate r/w
    return 2.0 * (input_bytes + hidden_bytes + num_layers * per_block)


def layers_for_row(row) -> int:
    # some timing CSVs have a 'layers' column, the older ones use layers == channels
    if "layers" in row and not pd.isna(row["layers"]):
        return int(row["layers"])
    return int(row["channels"])


# assemble all the points
def build_points() -> pd.DataFrame:
    records = []
    for src in TIMING_SOURCES:
        csv_path = src["csv"]
        if not csv_path.exists():
            print(f"WARNING: missing timing CSV, skipping: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        for row in df.to_dict("records"):
            B = int(row["batch"])
            T = int(row["tokens"])
            C = int(row["channels"])
            L = layers_for_row(row)
            flops = full_model_weighted_flops(
                build=src["build"], batch=B, tokens=T, channels_in=C, num_layers=L
            )
            bytes_est = estimated_bytes(batch=B, tokens=T, channels_in=C, num_layers=L)
            seconds = float(row["avg_ms"]) / 1000.0
            records.append({
                "impl": src["label"],
                "color": src["color"],
                "marker": src["marker"],
                "plot": src["plot"],
                "batch": B, "tokens": T, "channels": C, "layers": L,
                "weighted_flops": flops,
                "bytes_est": bytes_est,
                "intensity": flops / bytes_est,
                "gflops": flops / seconds / 1e9,
                "avg_ms": float(row["avg_ms"]),
            })
    return pd.DataFrame.from_records(records)


# --- plotting ---
BG = "#0f1117"
GRID = "#1e2130"
TEXT = "#e0e6f0"
MUTED = "#7a8ba0"
ROOF = "#ff8f00"          # scalar roof
ROOF_VEC = "#ab47bc"      # vectorized (SIMD) roof


def plot(points: pd.DataFrame, peak_gflops: float, bandwidth_gbps: float,
         simd_width: int, out_path: Path, dpi: int) -> None:
    plt.rcParams.update({
        "font.family": "monospace",
        "axes.facecolor": BG,
        "figure.facecolor": BG,
        "text.color": TEXT,
        "axes.labelcolor": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": GRID,
        "grid.color": GRID,
    })

    fig, ax = plt.subplots(figsize=(12, 7))

    peak_vec = peak_gflops * simd_width      # vectorized (NEON) compute ceiling

    plotted = points[points["plot"]].copy()
    ai = plotted["intensity"].to_numpy()
    ridge = peak_gflops / bandwidth_gbps          # scalar ridge
    ridge_vec = peak_vec / bandwidth_gbps         # vectorized ridge

    # pull x_min left enough to keep the ridges (and so the bandwidth diagonal)
    # in frame - all the measured points are way off to the right, compute-bound
    x_min = max(min(ai.min() / 3.0, ridge / 3.0), 1e-2)
    x_max = ai.max() * 3.0
    xs = np.logspace(np.log10(x_min), np.log10(x_max), 400)

    memory_roof = bandwidth_gbps * xs
    scalar_roof = np.minimum(memory_roof, np.full_like(xs, peak_gflops))
    vec_roof = np.minimum(memory_roof, np.full_like(xs, peak_vec))

    # shared memory roof (the bandwidth diagonal)
    ax.loglog(xs, memory_roof, color=MUTED, linestyle=":", alpha=0.6,
              label=f"Memory roof ({bandwidth_gbps:.0f} GB/s)")

    # vectorized roof (scalar x simd_width)
    ax.loglog(xs, vec_roof, color=ROOF_VEC, linewidth=2.5,
              label=f"Vectorized roof ({simd_width}x = {peak_vec:.1f} GFLOP/s)")
    ax.axvline(ridge_vec, color=ROOF_VEC, linestyle="--", linewidth=0.9, alpha=0.6)
    ax.text(ridge_vec * 1.05, peak_vec * 1.12, f"ridge {ridge_vec:.1f} F/B",
            color=ROOF_VEC, fontsize=7.0, va="center")

    # scalar roof
    ax.loglog(xs, scalar_roof, color=ROOF, linewidth=2.5,
              label=f"Scalar roof ({peak_gflops:.1f} GFLOP/s)")
    ax.axvline(ridge, color=ROOF, linestyle="--", linewidth=0.9, alpha=0.6)
    ax.text(ridge * 1.05, peak_gflops * 0.30, f"ridge {ridge:.1f} F/B",
            color=ROOF, fontsize=7.0, va="center")

    for impl, grp in plotted.groupby("impl", sort=False):
        ax.scatter(grp["intensity"], grp["gflops"],
                   color=grp["color"].iloc[0], marker=grp["marker"].iloc[0],
                   s=70, alpha=0.92, edgecolors="white", linewidths=0.5,
                   label=impl, zorder=4)
        for r in grp.itertuples(index=False):
            ax.annotate(f"T={r.tokens}", xy=(r.intensity, r.gflops),
                        xytext=(5, 4), textcoords="offset points",
                        fontsize=6.0, color=r.color, alpha=0.8)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity  [weighted FLOPs / estimated byte]",
                  fontsize=12, labelpad=10)
    ax.set_ylabel("Performance  [weighted GFLOP/s]", fontsize=12, labelpad=10)
    ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext())
    ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext())
    ax.set_xlim(x_min, x_max)

    # bump the top so the vectorized roof has headroom
    ax.set_ylim(plotted["gflops"].min() / 5.0, peak_vec * 2.5)

    ax.grid(which="both", linestyle="--", alpha=0.35)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)

    ax.set_title("Roofline — Apple M1 (single-core scalar & SIMD)",
                 fontsize=14, fontweight="bold", color=TEXT, pad=14)
    ax.legend(frameon=True, facecolor="#1a1f2e", edgecolor=GRID,
              labelcolor=TEXT, fontsize=8, loc="lower right",
              ncol=2, columnspacing=0.8, handletextpad=0.4)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved plot -> {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apple-M1 roofline for GATr kernels.")
    p.add_argument("--peak-gflops", type=float, default=25.6,
                   help="Single-core scalar FP32 peak in GFLOP/s (default 25.6).")
    p.add_argument("--bandwidth-gbps", type=float, default=60.0,
                   help="Single-core sustained bandwidth in GB/s (default 60).")
    p.add_argument("--simd-width", type=int, default=4,
                   help="SIMD width multiplier for the vectorized roof (default 4, M1 NEON).")
    p.add_argument("--out", type=Path,
                   default=RESULTS_DIR / "Roofline" / "Plots" / "m1_roofline.png")
    p.add_argument("--csv-out", type=Path,
                   default=RESULTS_DIR / "Roofline" / "CSVs" / "m1_roofline.csv")
    p.add_argument("--dpi", type=int, default=180)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    points = build_points()
    if points.empty:
        sys.exit("No data points generated (no timing CSVs found).")

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["impl", "batch", "tokens", "channels", "layers",
            "weighted_flops", "bytes_est", "intensity", "gflops", "avg_ms"]
    points[cols].to_csv(args.csv_out, index=False)
    print(f"Saved data -> {args.csv_out}")

    plot(points, args.peak_gflops, args.bandwidth_gbps, args.simd_width, args.out, args.dpi)

    # quick summary to stdout
    print(f"\n{'Impl':<26}{'T':>6}{'C':>5}{'L':>4}{'GFLOPs':>11}"
          f"{'AI(F/B)':>10}{'Perf GF/s':>12}")
    print("-" * 74)
    for r in points.itertuples(index=False):
        print(f"{r.impl:<26}{r.tokens:>6}{r.channels:>5}{r.layers:>4}"
              f"{r.weighted_flops/1e9:>11.4f}{r.intensity:>10.2f}{r.gflops:>12.3f}")


if __name__ == "__main__":
    main()
