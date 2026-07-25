"""
Sweeps the FLOP counter across the same (B, T, C) test cases as
baseline_c_code.py, emitting one row per (kernel, B, T, C) so the CSV
can be joined against c_per_kernel_cycles_results.csv for FLOPs/cycle.

Usage (from team80/):
    python Code/python/tools/plots/cpp_flop_counter_sweep.py
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TEAM80_DIR = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(SCRIPT_DIR.parent))

from cpp_flop_counter import (
    Build,
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


DATE = datetime.now().strftime("%Y-%m-%d")
RESULTS_DIR = TEAM80_DIR / "Results" / "timing_results" / "CSVs" / DATE

# One CSV per build, since the three builds compute different math.
BUILDS = [Build.BASELINE, Build.OPTIMIZED, Build.SIMD]


def output_file(build: Build) -> Path:
    return RESULTS_DIR / f"c_flops_results_{build.value}.csv"


# Match baseline_c_code.py's full sweep.
TEST_CASES = [(8, T, 2) for T in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]]

# same defaults inside CONFIG in cpp_flop_counter.py
CHANNELS_OUT = 32
CHANNELS_QK = 2
CHANNELS_V = 2
HEADS = 1
KINDS = ["ipa", "daa"]

WEIGHTS = OpWeights()


def per_case_reports(B: int, T: int, C: int, build: Build):
    gp_m = B * T * C
    join_m = gp_m
    gelu_m = B * T * CHANNELS_OUT
    norm_n = B * T
    norm_c = CHANNELS_OUT

    return {
        "equi_linear": equi_linear_count(
            EquiLinearCfg(batch=B, tokens=T, channels_in=C, channels_out=CHANNELS_OUT, bias=True, build=build)
        ),
        "equi_rms_norm": equi_rms_norm_count(
            EquiRMSNormCfg(n=norm_n, channels=norm_c, hasWt=True, mOne=False, build=build)
        ),
        "geometric_product": geometric_product_count(GeometricProductCfg(m=gp_m, build=build)),
        "equi_join": equi_join_count(
            EquiJoinCfg(m=join_m, reference=True, x_nnz_frac=1.0, include_kernel_build=False, noCompOpt=True, build=build)
        ),
        "equi_geometric_attention": equi_geometric_attention_count(
            EquiGeometricAttentionCfg(
                batch=B, heads=HEADS, t_query=T, t_key=T,
                channels_qk=CHANNELS_QK, channels_v=CHANNELS_V,
                kinds=KINDS, build=build,
            )
        ),
        "scaler_gated_gelu": scaler_gated_gelu_count(ScalerGatedGeluCfg(m=gelu_m, build=build)),
    }


def write_sweep(build: Build) -> Path:
    out = output_file(build)
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "build", "kernel", "batch", "tokens", "channels",
            "add", "mul", "div", "sqrt", "exp", "tanh", "pow2",
            "flops_total", "weighted_flops",
        ])

        for B, T, C in TEST_CASES:
            reports = per_case_reports(B, T, C, build)
            total_flops = 0
            for kname, tally in reports.items():
                writer.writerow([
                    build.value, kname, B, T, C,
                    int(tally.add), int(tally.mul), int(tally.div),
                    int(tally.sqrt), int(tally.exp), int(tally.tanh), int(tally.pow2),
                    int(tally.flops_total()),
                    int(tally.weighted_flops(WEIGHTS)),
                ])
                total_flops += tally.flops_total()
            print(f"    B={B}, T={T}, C={C}: {int(total_flops):,} total flops across {len(reports)} kernels")
    return out


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for build in BUILDS:
        print(f"\n[{build.value}] writing {output_file(build)}")
        write_sweep(build)


if __name__ == "__main__":
    main()
