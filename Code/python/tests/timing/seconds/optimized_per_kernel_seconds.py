"""
Per-kernel timing in seconds for C++ or pure-Python implementations.

Times each kernel directly with time.perf_counter_ns, using the same
methodology as baseline_c_code.py / optimized_c_code.py. No C++
counter instrumentation needed.

Run with:
    python -m tests.timing.seconds.optimized_per_kernel_seconds --impl baseline
    python -m tests.timing.seconds.optimized_per_kernel_seconds --impl optimized
    python -m tests.timing.seconds.optimized_per_kernel_seconds --impl optimized_for_cache
    python -m tests.timing.seconds.optimized_per_kernel_seconds --impl simd
    python -m tests.timing.seconds.optimized_per_kernel_seconds --impl python

Writes
Results/timing_results/CSVs/per_kernel_<DATE>/<impl>_per_kernel_seconds_results_<run-id>.csv
with one row per (kernel, B, T, C).
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
TEAM80_DIR = SCRIPT_DIR.parents[4]
CODE_DIR = SCRIPT_DIR.parents[2]

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

DATE = datetime.now().strftime("%Y-%m-%d")
DATE_TIME = os.environ.get("TIMING_RUN_ID", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

WARMUP = 10
RUNS = 30
SEED = 42

# Per-kernel sweep. Same shape as the full-forward sweep so kernels see
# tensors of the same size they'd see in a real forward pass.
BATCH = 8
BASE_C, BASE_T, BASE_L = 1, 2, 1
N_VALUES = [1, 2, 4, 8, 16, 32, 64, 128]
TEST_CASES = [(BATCH, BASE_T*N, BASE_C*N, BASE_L*N) for N in N_VALUES]

# Maps the --impl flag to (CPP_IMPL value, IMPLEMENTATION_NAME, file prefix).
# CPP_IMPL is None for the pure-Python path (no build step).
IMPL_CONFIG = {
    "baseline":            ("baseline",            "C++",                 "baseline_c"),
    "optimized":           ("optimized",           "Optimized C++",       "optimized_c"),
    "optimized_for_cache": ("optimized_for_cache", "Optimized-for-cache C++", "optimized_for_cache_c"),
    "simd":                ("simd",                "Vectorized C++",      "simd_c"),
    "python":              (None,                   "Python",              "python"),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--impl",
        choices=list(IMPL_CONFIG.keys()),
        required=True,
        help="Which C++ implementation to build and benchmark.",
    )
    return parser.parse_args()


def build_cpp_impl(cpp_impl):
    for pattern in ("*.so", "*.pyd", "*.dll"):
        for path in (CODE_DIR / "ops").glob(pattern):
            path.unlink()

    env = os.environ.copy()
    env["CPP_IMPL"] = cpp_impl
    subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace", "--force"],
        cwd=CODE_DIR,
        env=env,
        check=True,
    )


def time_call(fn, args, repeats):
    """Same timing pattern as baseline_c_code.py.time_model."""
    with torch.no_grad():
        for _ in range(WARMUP):
            _ = fn(*args)

    times_ms = []
    with torch.no_grad():
        for _ in range(repeats):
            start = time.perf_counter_ns()
            _ = fn(*args)
            end = time.perf_counter_ns()
            times_ms.append((end - start) / 1e6)

    return (
        sum(times_ms) / len(times_ms),
        min(times_ms),
        max(times_ms),
    )


def call_equi_geometric_attention(fn, query, key, value, kinds, weight):
    return fn(
        query,
        key,
        value,
        kinds=kinds,
        weight=weight,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=True,
        scale=None,
    )


def build_inputs(batch, tokens, channels, mods):
    """Generate random inputs of the right shape for each kernel."""
    ops_linear, ops_dual, ops_norm, ops_attention = mods

    x = torch.randn(batch, tokens, channels, 16)
    y = torch.randn(batch, tokens, channels, 16)
    ref = torch.randn(batch, tokens, channels, 16)

    # equi_linear inputs
    weight = torch.randn(channels, channels, 9)
    basis = torch.randn(9, 16, 16)
    bias = torch.randn(channels)

    # equi_rms_norm weight
    rms_weight = torch.randn(channels)

    # equi_geometric_attention inputs: MV-only path, (B, H, T, C, 16).
    HEADS = 4
    q_att = torch.randn(batch, HEADS, tokens, channels, 16)
    k_att = torch.randn(batch, HEADS, tokens, channels, 16)
    v_att = torch.randn(batch, HEADS, tokens, channels, 16)
    att_kinds = {"ipa": None, "daa": None}
    # Neutral attention mixing. Use scalar tensors because the SIMD/optimized
    # C++ kernels squeeze weights before expanding them to (heads, channels);
    # shaped all-ones weights become ambiguous when channels == 1.
    att_weight = [torch.tensor(1.0), torch.tensor(1.0)]

    # Use the packed path for equi_linear if available — that is what the model
    # uses at runtime (weights are packed once at init, then equi_linear_packed
    # is called every forward pass).
    if hasattr(ops_linear, "pack_equi_linear_weight") and hasattr(ops_linear, "equi_linear_packed"):
        packed_weight = ops_linear.pack_equi_linear_weight(weight)
        equi_linear_fn   = ops_linear.equi_linear_packed
        equi_linear_args = (x, packed_weight, basis, bias)
    else:
        equi_linear_fn   = ops_linear.equi_linear
        equi_linear_args = (x, weight, basis, bias)

    return [
        ("geometric_product",        ops_linear.geometric_product,        (x, y)),
        ("equi_linear",              equi_linear_fn,                      equi_linear_args),
        ("equi_join",                ops_dual.equi_join,                  (x, y, ref)),
        ("equi_rms_norm",            ops_norm.equi_rms_norm,              (x, rms_weight)),
        ("scaler_gated_gelu",        ops_norm.scaler_gated_gelu,          (x,)),
        (
            "equi_geometric_attention",
            lambda query, key, value, kinds, weight: call_equi_geometric_attention(
                ops_attention.equi_geometric_attention,
                query,
                key,
                value,
                kinds,
                weight,
            ),
            (q_att, k_att, v_att, att_kinds, att_weight),
        ),
    ]


def build_inputs_python(batch, tokens, channels):
    """Build inputs for the pure-Python ezgatr kernel functions."""
    from ezgatr.nn.functional.linear import geometric_product, equi_linear
    from ezgatr.nn.functional.dual import equi_join
    from ezgatr.nn.functional.norm import equi_rms_norm
    from ezgatr.nn.functional.activation import scaler_gated_gelu
    from ezgatr.nn.functional import equi_geometric_attention

    x = torch.randn(batch, tokens, channels, 16)
    y = torch.randn(batch, tokens, channels, 16)
    ref = torch.randn(batch, tokens, channels, 16)

    # equi_linear: Python signature is (x, weight, bias) — no explicit basis arg.
    weight = torch.randn(channels, channels, 9)
    bias = torch.randn(channels)
    rms_weight = torch.randn(channels)

    HEADS = 4
    q_att = torch.randn(batch, HEADS, tokens, channels, 16)
    k_att = torch.randn(batch, HEADS, tokens, channels, 16)
    v_att = torch.randn(batch, HEADS, tokens, channels, 16)
    att_kinds = {"ipa": None, "daa": None}
    att_weight = [torch.tensor(1.0), torch.tensor(1.0)]

    return [
        ("geometric_product", geometric_product, (x, y)),
        ("equi_linear",       equi_linear,       (x, weight, bias)),
        ("equi_join",         equi_join,          (x, y, ref)),
        ("equi_rms_norm",     equi_rms_norm,      (x, rms_weight)),
        ("scaler_gated_gelu", scaler_gated_gelu,  (x,)),
        (
            "equi_geometric_attention",
            lambda query, key, value, kinds, weight: call_equi_geometric_attention(
                equi_geometric_attention,
                query,
                key,
                value,
                kinds,
                weight,
            ),
            (q_att, k_att, v_att, att_kinds, att_weight),
        ),
    ]


def benchmark(cpp_impl, impl_name, file_prefix):
    # Some C++ kernels load basis/*.pt through a relative path. Keep the
    # process cwd at Code/python while writing results to the top-level Results
    # directory through absolute paths.
    os.chdir(CODE_DIR)

    output_file = (
        TEAM80_DIR / "Results" / "timing_results" / "CSVs" / f"per_kernel_{DATE}"
        / f"{file_prefix}_per_kernel_seconds_results_{DATE_TIME}.csv"
    )

    print("\033[35m========================================\033[0m")
    print(f"\033[35m  {impl_name} Per-Kernel Timing [ms]\033[0m")
    print("\033[35m========================================\033[0m\n")

    is_python = cpp_impl is None
    src_info = "ezgatr.nn.functional" if is_python else f"cpp/{cpp_impl}"
    print(f"\033[36m[INFO]\033[0m Implementation: {impl_name}")
    print(f"\033[36m[INFO]\033[0m Source: {src_info}")
    print(f"\033[36m[INFO]\033[0m Warmup runs: {WARMUP}")
    print(f"\033[36m[INFO]\033[0m Timed repeats: {RUNS}")
    print(f"\033[36m[INFO]\033[0m Test cases: {len(TEST_CASES)}")
    print(f"\033[36m[INFO]\033[0m Output file: {output_file}\n")

    if not is_python:
        # Import the freshly built ops modules after the build step so we
        # pick up the right .so files.
        from ops import linear as ops_linear
        from ops import dual as ops_dual
        from ops import norm as ops_norm
        from ops import attention as ops_attention
        mods = (ops_linear, ops_dual, ops_norm, ops_attention)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_file.exists()
    with output_file.open("a", newline="") as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow([
                "implementation",
                "kernel",
                "batch",
                "tokens",
                "channels",
                "layers",
                "avg_ms",
                "min_ms",
                "max_ms",
            ])

        for batch, tokens, channels, layers in TEST_CASES:
            torch.manual_seed(SEED)

            print(
                f"\033[34m[STEP]\033[0m B={batch}, T={tokens}, C={channels}, L={layers}",
                flush=True,
            )

            kernels = (
                build_inputs_python(batch, tokens, channels)
                if is_python
                else build_inputs(batch, tokens, channels, mods)
            )

            for kname, fn, args in kernels:
                avg_ms, min_ms, max_ms = time_call(fn, args, RUNS)

                writer.writerow([
                    impl_name,
                    kname,
                    batch,
                    tokens,
                    channels,
                    layers,
                    avg_ms,
                    min_ms,
                    max_ms,
                ])
                f.flush()

                print(
                    f"\033[32m[{kname}]\033[0m "
                    f"avg={avg_ms:.4f} ms | min={min_ms:.4f} ms | max={max_ms:.4f} ms"
                )
            print()

    print(f"\033[36m[INFO]\033[0m Saved results to {output_file}")
    print("\n\033[32m========================================\033[0m")
    print(f"\033[32m  {impl_name} per-kernel benchmark completed\033[0m")
    print("\033[32m========================================\033[0m")


if __name__ == "__main__":
    args = parse_args()
    cpp_impl, impl_name, file_prefix = IMPL_CONFIG[args.impl]
    if cpp_impl is not None:
        build_cpp_impl(cpp_impl)
    benchmark(cpp_impl, impl_name, file_prefix)
