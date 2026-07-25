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

RESULTS_DIR = TEAM80_DIR / "Results" / "timing_results" / "CSVs" / DATE
OUTPUT_FILE = RESULTS_DIR / f"optimized_c_timing_results_{DATE_TIME}.csv"
CPP_IMPL = "optimized"

WARMUP = 10
RUNS = 30
SEED = 42


IMPLEMENTATION_NAME = "Optimized C++"

BATCH = 8
BASE_C, BASE_T, BASE_L = 1, 2, 1
N_VALUES = [1, 2, 4, 8, 16, 32, 64, 128]
TEST_CASES = [(BATCH, BASE_T*N, BASE_C*N, BASE_L*N) for N in N_VALUES]


def build_cpp_impl():
    for pattern in ("*.so", "*.pyd", "*.dll"):
        for path in (CODE_DIR / "ops").glob(pattern):
            path.unlink()

    env = os.environ.copy()
    env["CPP_IMPL"] = CPP_IMPL
    subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace", "--force"],
        cwd=CODE_DIR,
        env=env,
        check=True,
    )


build_cpp_impl()

from ezgatr.nets.mv_only_gatr import MVOnlyGATrConfig
from ezgatr_extensions.asl_mv_only_gatr import ASLMVOnlyGATrModel


def get_repeats(T):
    return RUNS
  


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

    avg_ms = sum(times_ms) / len(times_ms)
    min_ms = min(times_ms)
    max_ms = max(times_ms)

    return avg_ms, min_ms, max_ms


def benchmark_optimized_cpp():
    os.chdir(CODE_DIR)

    print("\033[35m========================================\033[0m")
    print("\033[35m  Optimized C++ Timing Benchmark [ms]\033[0m")
    print("\033[35m========================================\033[0m\n")

    print(f"\033[36m[INFO]\033[0m Implementation: {IMPLEMENTATION_NAME}")
    print(f"\033[36m[INFO]\033[0m C++ source folder: cpp/{CPP_IMPL}")
    print(f"\033[36m[INFO]\033[0m Warmup runs: {WARMUP}")
    print(f"\033[36m[INFO]\033[0m Timed repeats: adaptive (10/5/2 by T)")
    print(f"\033[36m[INFO]\033[0m Test cases: {len(TEST_CASES)}")
    print(f"\033[36m[INFO]\033[0m Output file: {OUTPUT_FILE}\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    write_header = not OUTPUT_FILE.exists()

    with OUTPUT_FILE.open("a", newline="") as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow([
                "implementation",
                "batch",
                "tokens",
                "channels",
                "layers",
                "avg_ms",
                "min_ms",
                "max_ms",
            ])

        for batch, tokens, number_of_channels, layers in TEST_CASES:
            torch.manual_seed(SEED)

            print(
                f"\033[34m[STEP]\033[0m Running B={batch}, "
                f"T={tokens}, C={number_of_channels}, L={layers}",
                flush=True,
            )

            config = MVOnlyGATrConfig(size_channels_in=number_of_channels, num_layers = layers)
            cpp_net = ASLMVOnlyGATrModel(config)

            x = torch.randn(batch, tokens, number_of_channels, 16)

            repeats = get_repeats(tokens)
            avg_ms, min_ms, max_ms = time_model(cpp_net, x, repeats)

            writer.writerow([
                IMPLEMENTATION_NAME,
                batch,
                tokens,
                number_of_channels,
                layers,  
                avg_ms,
                min_ms,
                max_ms,
            ])

            print(
                f"\033[32m[{IMPLEMENTATION_NAME}]\033[0m "
                f"B={batch}, T={tokens}, C={number_of_channels}, L={layers} | "
                f"avg={avg_ms:.3f} ms | "
                f"min={min_ms:.3f} ms | "
                f"max={max_ms:.3f} ms\n"
            )

    print(f"\n\033[36m[INFO]\033[0m Saved {IMPLEMENTATION_NAME} results to {OUTPUT_FILE}")

    print("\n\033[32m========================================\033[0m")
    print("\033[32m  Optimized C++ timing benchmark completed\033[0m")
    print("\033[32m========================================\033[0m")


if __name__ == "__main__":
    benchmark_optimized_cpp()
