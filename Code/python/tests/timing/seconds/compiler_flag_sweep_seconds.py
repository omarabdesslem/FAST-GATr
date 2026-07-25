"""
Timing sweep for optimized vs SIMD C++ builds under different compiler flags.

This script is intended for Apple Silicon/macOS. It compares the optimized and
SIMD implementations with clang flag presets that matter on ARM64/NEON:
optimization level, native CPU tuning, fast math, loop unrolling, and clang
auto-vectorization.

Run from the repository root or Code/python:
    python Code/python/tests/timing/seconds/compiler_flag_sweep_seconds.py

Writes:
Results/timing_results/CSVs/flag_sweep_<DATE>/optimized_vs_simd_flag_sweep_<run-id>.csv
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

RESULTS_DIR = TEAM80_DIR / "Results" / "timing_results" / "CSVs" / f"flag_sweep_{DATE}"
OUTPUT_FILE = RESULTS_DIR / f"optimized_vs_simd_flag_sweep_{DATE_TIME}.csv"

WARMUP = 10
RUNS = 30
SEED = 42

BATCH = 8
BASE_C, BASE_T, BASE_L = 1, 2, 1
N_VALUES = [1, 2, 4, 8, 16, 32, 64]
TEST_CASES = [(BATCH, BASE_T * N, BASE_C * N, BASE_L * N) for N in N_VALUES]

IMPLEMENTATIONS = {
    "simd": "SIMD C++",
    "optimized": "Optimized C++",
}

MAC_WARNING_FLAGS = "-Wno-error=invalid-specialization -Wno-invalid-specialization"

FLAG_PRESETS = [
    (
        "default_simd",
        f"-std=c++17 -O3 -DNDEBUG -funroll-loops {MAC_WARNING_FLAGS} -mcpu=native",
    ),
    (
        "o2_no_autovec",
        f"-std=c++17 -Wall -O2 -DNDEBUG -fno-vectorize -fno-slp-vectorize {MAC_WARNING_FLAGS} -mcpu=native",
    ),
    (
        "o3_no_autovec",
        f"-std=c++17 -Wall -O3 -DNDEBUG -fno-vectorize -fno-slp-vectorize {MAC_WARNING_FLAGS} -mcpu=native",
    ),
    (
        "o3_native",
        f"-std=c++17 -Wall -O3 -DNDEBUG {MAC_WARNING_FLAGS} -mcpu=native",
    ),
    (
        "o3_native_fast_math",
        f"-std=c++17 -Wall -O3 -DNDEBUG -ffast-math {MAC_WARNING_FLAGS} -mcpu=native",
    ),
    (
        "o3_native_fast_math_unroll",
        f"-std=c++17 -Wall -O3 -DNDEBUG -ffast-math -funroll-loops {MAC_WARNING_FLAGS} -mcpu=native",
    ),
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--impl", choices=IMPLEMENTATIONS, help=argparse.SUPPRESS)
    parser.add_argument("--flag-id", help=argparse.SUPPRESS)
    parser.add_argument("--flags", help=argparse.SUPPRESS)
    parser.add_argument("--output-file", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def clean_ops():
    for pattern in ("*.so", "*.pyd", "*.dll"):
        for path in (CODE_DIR / "ops").glob(pattern):
            path.unlink()


def build_cpp_impl(cpp_impl, flags):
    clean_ops()
    env = os.environ.copy()
    env["CPP_IMPL"] = cpp_impl
    env["ASL_CXX_FLAGS"] = flags
    subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace", "--force"],
        cwd=CODE_DIR,
        env=env,
        check=True,
    )


def time_model(model, x, repeats):
    model.eval()

    with torch.no_grad():
        for _ in range(WARMUP):
            _ = model(x)

    times_ms = []
    with torch.no_grad():
        for _ in range(repeats):
            start = time.perf_counter_ns()
            _ = model(x)
            end = time.perf_counter_ns()
            times_ms.append((end - start) / 1e6)

    return sum(times_ms) / len(times_ms), min(times_ms), max(times_ms)


def append_header_if_needed(output_file):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists():
        return
    with output_file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "implementation",
            "flag_id",
            "flags",
            "batch",
            "tokens",
            "channels",
            "layers",
            "avg_ms",
            "min_ms",
            "max_ms",
        ])


def has_existing_o2_result(impl_name):
    if not RESULTS_DIR.exists():
        return False

    expected_tokens = {BASE_T * N for N in N_VALUES}
    seen_tokens = set()

    for csv_file in RESULTS_DIR.glob("optimized_vs_simd_flag_sweep_*.csv"):
        with csv_file.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("implementation") != impl_name:
                    continue
                if row.get("flag_id") != "o2_no_autovec":
                    continue
                try:
                    seen_tokens.add(int(row["tokens"]))
                except (KeyError, ValueError):
                    continue

    return expected_tokens.issubset(seen_tokens)


def run_worker(cpp_impl, impl_name, flag_id, flags, output_file):
    os.chdir(CODE_DIR)
    print(f"\033[36m[INFO]\033[0m Building {impl_name} with {flag_id}")
    print(f"\033[36m[INFO]\033[0m Flags: {flags}")
    build_cpp_impl(cpp_impl, flags)

    from ezgatr.nets.mv_only_gatr import MVOnlyGATrConfig
    from ezgatr_extensions.asl_mv_only_gatr import ASLMVOnlyGATrModel

    append_header_if_needed(output_file)

    with output_file.open("a", newline="") as f:
        writer = csv.writer(f)

        for batch, tokens, channels, layers in TEST_CASES:
            torch.manual_seed(SEED)
            config = MVOnlyGATrConfig(size_channels_in=channels, num_layers=layers)
            model = ASLMVOnlyGATrModel(config)
            x = torch.randn(batch, tokens, channels, 16)

            avg_ms, min_ms, max_ms = time_model(model, x, RUNS)
            writer.writerow([
                impl_name,
                flag_id,
                flags,
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
                f"\033[32m[{impl_name} | {flag_id}]\033[0m "
                f"B={batch}, T={tokens}, C={channels}, L={layers} | "
                f"avg={avg_ms:.3f} ms | min={min_ms:.3f} ms | max={max_ms:.3f} ms",
                flush=True,
            )


def run_sweep(output_file):
    append_header_if_needed(output_file)

    print("\033[35m========================================\033[0m")
    print("\033[35m  Optimized vs SIMD Compiler Flag Sweep\033[0m")
    print("\033[35m========================================\033[0m\n")
    print(f"\033[36m[INFO]\033[0m N values: {N_VALUES}")
    print(f"\033[36m[INFO]\033[0m Warmup runs: {WARMUP}")
    print(f"\033[36m[INFO]\033[0m Timed repeats: {RUNS}")
    print(f"\033[36m[INFO]\033[0m Output file: {output_file}\n")

    for cpp_impl, impl_name in IMPLEMENTATIONS.items():
        for flag_id, flags in FLAG_PRESETS:
            if flag_id == "default_simd" and cpp_impl != "simd":
                continue
            if flag_id == "o2_no_autovec" and has_existing_o2_result(impl_name):
                print(
                    f"\033[33m[SKIP]\033[0m {impl_name} | {flag_id} "
                    f"already has today's N<=64 results",
                    flush=True,
                )
                continue
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "compiler_flag_sweep_seconds.py"),
                    "--worker",
                    "--impl",
                    cpp_impl,
                    "--flag-id",
                    flag_id,
                    "--flags",
                    flags,
                    "--output-file",
                    str(output_file),
                ],
                cwd=TEAM80_DIR,
                check=True,
            )

    print(f"\n\033[36m[INFO]\033[0m Saved flag sweep results to {output_file}")


if __name__ == "__main__":
    args = parse_args()
    if args.worker:
        run_worker(
            args.impl,
            IMPLEMENTATIONS[args.impl],
            args.flag_id,
            args.flags,
            args.output_file,
        )
    else:
        run_sweep(OUTPUT_FILE)
