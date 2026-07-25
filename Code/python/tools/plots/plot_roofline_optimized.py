#!/usr/bin/env python3
"""
Estimated roofline plot for the optimized C++ timing results.

This is an estimated roofline, not a measured memory-traffic roofline:
- FLOPs come from Code/python/tools/cpp_flop_counter.py.
- Runtime comes from the optimized timing CSV.
- Bytes are a conservative tensor-traffic estimate, because we do not have
  measured DRAM bytes from hardware counters in the timing CSV.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
TEAM80_DIR = SCRIPT_DIR.parents[3]
TOOLS_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(TOOLS_DIR))

from cpp_flop_counter import (  # noqa: E402
    EquiGeometricAttentionCfg,
    EquiJoinCfg,
    EquiLinearCfg,
    EquiRMSNormCfg,
    GeometricProductCfg,
    OpWeights,
    ScalerGatedGeluCfg,
    equi_geometric_attention_count,
    equi_join_count,
    equi_linear_count,
    equi_rms_norm_count,
    geometric_product_count,
    scaler_gated_gelu_count,
)


DEFAULT_TIMING = (
    TEAM80_DIR
    / "Results"
    / "timing_results"
    / "CSVs"
    / "2026-05-20"
    / "optimized_c_timing_results_2026-05-20_15-37-13.csv"
)
DEFAULT_OUT = TEAM80_DIR / "Results" / "Roofline" / "Plots" / "optimized_roofline.png"

CHANNELS_OUT = 32
CHANNELS_QK = 2
CHANNELS_V = 2
HEADS = 1
KINDS = ["ipa", "daa"]
FLOAT_BYTES = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimated roofline for optimized C++ timing results.")
    parser.add_argument("--timing", type=Path, default=DEFAULT_TIMING, help="Optimized timing CSV.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output PNG path.")
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=TEAM80_DIR / "Results" / "Roofline" / "CSVs" / "optimized_roofline.csv",
        help="Output CSV path for the plotted roofline points.",
    )
    parser.add_argument(
        "--peak-gflops",
        type=float,
        default=2600.0,
        help="Assumed FP32 compute peak in GFLOP/s. Default is an Apple-M1-scale placeholder.",
    )
    parser.add_argument(
        "--bandwidth-gbps",
        type=float,
        default=68.0,
        help="Assumed memory bandwidth in GB/s. Default is an Apple-M1-scale placeholder.",
    )
    return parser.parse_args()


def weighted_flops_for_case(batch: int, tokens: int, channels: int) -> float:
    weights = OpWeights()
    gp_m = batch * tokens * channels
    join_m = gp_m
    gelu_m = batch * tokens * CHANNELS_OUT
    norm_n = batch * tokens
    norm_c = CHANNELS_OUT

    counts = [
        equi_linear_count(
            EquiLinearCfg(
                batch=batch,
                tokens=tokens,
                channels_in=channels,
                channels_out=CHANNELS_OUT,
                bias=True,
            )
        ),
        equi_rms_norm_count(EquiRMSNormCfg(n=norm_n, channels=norm_c, hasWt=True, mOne=False)),
        geometric_product_count(GeometricProductCfg(m=gp_m)),
        equi_join_count(
            EquiJoinCfg(
                m=join_m,
                reference=True,
                x_nnz_frac=1.0,
                include_kernel_build=False,
                noCompOpt=True,
            )
        ),
        equi_geometric_attention_count(
            EquiGeometricAttentionCfg(
                batch=batch,
                heads=HEADS,
                t_query=tokens,
                t_key=tokens,
                channels_qk=CHANNELS_QK,
                channels_v=CHANNELS_V,
                kinds=KINDS,
            )
        ),
        scaler_gated_gelu_count(ScalerGatedGeluCfg(m=gelu_m)),
    ]
    return sum(c.weighted_flops(weights) for c in counts)


def estimated_bytes_for_case(batch: int, tokens: int, channels: int) -> float:
    """Estimate compulsory tensor traffic for the main tensors.

    This intentionally stays simple and transparent. It includes input/output
    tensors and large attention matrices, then multiplies by 2 to account for
    intermediate reads/writes across fused and unfused kernels. It is not a
    substitute for measured DRAM bytes from hardware counters.
    """
    input_bytes = batch * tokens * channels * 16 * FLOAT_BYTES
    hidden_bytes = batch * tokens * CHANNELS_OUT * 16 * FLOAT_BYTES
    qkv_bytes = 3 * batch * HEADS * tokens * CHANNELS_OUT * 16 * FLOAT_BYTES
    attention_matrix_bytes = batch * HEADS * tokens * tokens * FLOAT_BYTES
    output_bytes = hidden_bytes
    return 2.0 * (input_bytes + hidden_bytes + qkv_bytes + attention_matrix_bytes + output_bytes)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.timing)

    rows = []
    for row in df.itertuples(index=False):
        flops = weighted_flops_for_case(int(row.batch), int(row.tokens), int(row.channels))
        bytes_est = estimated_bytes_for_case(int(row.batch), int(row.tokens), int(row.channels))
        seconds = float(row.avg_ms) / 1_000.0
        rows.append(
            {
                "batch": int(row.batch),
                "tokens": int(row.tokens),
                "channels": int(row.channels),
                "flops": flops,
                "bytes_est": bytes_est,
                "intensity": flops / bytes_est,
                "gflops": flops / seconds / 1e9,
            }
        )
    roof = pd.DataFrame(rows)

    x_min = max(roof["intensity"].min() / 2.0, 1e-3)
    x_max = roof["intensity"].max() * 2.0
    xs = np.logspace(np.log10(x_min), np.log10(x_max), 256)
    memory_roof = args.bandwidth_gbps * xs
    compute_roof = np.full_like(xs, args.peak_gflops)
    roofline = np.minimum(memory_roof, compute_roof)
    ridge = args.peak_gflops / args.bandwidth_gbps

    background = "#0f1117"
    grid_color = "#1e2130"
    text_color = "#e0e6f0"
    muted = "#7a8ba0"
    accent = "#4fc3f7"
    roof_color = "#ffb86c"

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
        }
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(xs, roofline, color=roof_color, linewidth=2.5, label="Roofline")
    ax.loglog(xs, memory_roof, color=roof_color, linestyle=":", alpha=0.6, label="Memory roof")
    ax.loglog(xs, compute_roof, color=roof_color, linestyle="--", alpha=0.6, label="Compute roof")
    ax.axvline(ridge, color=muted, linestyle="--", linewidth=1.0, alpha=0.7)

    ax.scatter(roof["intensity"], roof["gflops"], color=accent, s=58, zorder=3, label="Optimized C++")
    for row in roof.itertuples(index=False):
        ax.annotate(
            f"T={row.tokens}",
            xy=(row.intensity, row.gflops),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=8,
            color=accent,
        )

    ax.set_title("Estimated Roofline: Optimized C++", fontsize=14, fontweight="bold", pad=16)
    ax.set_xlabel("Operational intensity: weighted FLOPs / estimated byte")
    ax.set_ylabel("Achieved performance: weighted GFLOP/s")
    ax.grid(True, which="both", linestyle="--", alpha=0.45)
    ax.legend(
        frameon=True,
        facecolor="#1a1f2e",
        edgecolor=grid_color,
        labelcolor=text_color,
        fontsize=9,
        loc="lower right",
    )

    note = (
        f"Assumed peak={args.peak_gflops:.0f} GFLOP/s, "
        f"bandwidth={args.bandwidth_gbps:.0f} GB/s; bytes are estimated."
    )
    ax.text(0.01, 0.02, note, transform=ax.transAxes, color=muted, fontsize=8)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=180, bbox_inches="tight")
    roof.to_csv(args.csv_out, index=False)
    print(f"Saved -> {args.out}")
    print(f"Saved data -> {args.csv_out}")
    plt.show()


if __name__ == "__main__":
    main()
