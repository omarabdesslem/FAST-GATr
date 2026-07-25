"""
Cycle-count benchmark for the C++ implementation.

Mirrors baseline_c_code.py but measures full-forward-pass cycles using
__rdtsc instead of ms.
Writes to Results/timing_results/CSVs/<DATE>/c_cycles_results.csv.

Build the helper DLL once:
    cl /LD /O2 /nologo _rdtsc_helper.c /Fe:_rdtsc_helper.dll

Run with:
    python -m tests.timing.cycles.baseline_c_cycles
"""

import csv
import ctypes
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
TEAM80_DIR = SCRIPT_DIR.parents[4]
CODE_DIR = SCRIPT_DIR.parents[2]

DATE = datetime.now().strftime("%Y-%m-%d")
DATE_TIME = os.environ.get("TIMING_RUN_ID", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

RESULTS_DIR = TEAM80_DIR / "Results" / "timing_results" / "CSVs" / DATE
OUTPUT_FILE = RESULTS_DIR / "c_cycles_results.csv"
RUN_FILE = RESULTS_DIR / f"c_cycles_results_{DATE_TIME}.csv"
CPP_IMPL = "baseline"

WARMUP = 10
SEED = 42

IMPLEMENTATION_NAME = "C++"

BATCH = 8
BASE_C, BASE_T, BASE_L = 2, 64, 16
N_VALUES = [1, 2, 4, 8, 16]  
TEST_CASES = [(BATCH, BASE_T*N, BASE_C*N, BASE_L*N) for N in N_VALUES]


REPEATS = 30


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


def load_rdtsc():
    dll_path = SCRIPT_DIR / "_rdtsc_helper.dll"
    if not dll_path.exists():
        print(
            f"\033[31m[ERROR]\033[0m DLL not found: {dll_path}\n"
            "Build it first:\n"
            f"  cd {SCRIPT_DIR}\n"
            "  cl /LD /O2 /nologo _rdtsc_helper.c /Fe:_rdtsc_helper.dll",
            file=sys.stderr,
        )
        sys.exit(1)

    lib = ctypes.CDLL(str(dll_path))
    lib.rdtsc_start.restype = ctypes.c_uint64
    lib.rdtsc_start.argtypes = []
    lib.rdtsc_end.restype = ctypes.c_uint64
    lib.rdtsc_end.argtypes = []
    return lib


def cycles_model(model, x, rdtsc, repeats):
    model.eval()

    with torch.no_grad():
        for _ in range(WARMUP):
            _ = model(x)

    cycles = []

    with torch.no_grad():
        for _ in range(repeats):
            start = rdtsc.rdtsc_start()
            _ = model(x)
            end = rdtsc.rdtsc_end()
            cycles.append(end - start)

    avg = sum(cycles) / len(cycles)
    return avg, min(cycles), max(cycles)


def benchmark_cycles():
    print("\033[35m========================================\033[0m")
    print("\033[35m  EzGATr ASL/C++ Cycle Benchmark (RDTSC)\033[0m")
    print("\033[35m========================================\033[0m\n")

    rdtsc = load_rdtsc()

    print(f"\033[36m[INFO]\033[0m Implementation: {IMPLEMENTATION_NAME}")
    print(f"\033[36m[INFO]\033[0m C++ source folder: cpp/{CPP_IMPL}")
    print(f"\033[36m[INFO]\033[0m Warmup runs: {WARMUP}")
    print(f"\033[36m[INFO]\033[0m Timed repeats: {REPEATS}")
    print(f"\033[36m[INFO]\033[0m Test cases: {len(TEST_CASES)}")
    print(f"\033[36m[INFO]\033[0m Output file: {OUTPUT_FILE}")
    print(f"\033[36m[INFO]\033[0m Per-run file: {RUN_FILE}\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    f_main = OUTPUT_FILE.open("w", newline="")
    f_run = RUN_FILE.open("w", newline="")
    try:
        writer_main = csv.writer(f_main)
        writer_run = csv.writer(f_run)

        header = [
            "implementation",
            "batch",
            "tokens",
            "channels",
            "layers",
            "avg_cycles",
            "min_cycles",
            "max_cycles",
        ]
        writer_main.writerow(header)
        writer_run.writerow(header)

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

            avg, mn, mx = cycles_model(cpp_net, x, rdtsc, REPEATS)

            row = [
                IMPLEMENTATION_NAME,
                batch,
                tokens,
                number_of_channels,
                layers,  
                int(avg),
                int(mn),
                int(mx),
            ]
            writer_main.writerow(row)
            writer_run.writerow(row)
            f_main.flush()
            f_run.flush()

            print(
                f"\033[32m[{IMPLEMENTATION_NAME}]\033[0m "
                f"B={batch}, T={tokens}, C={number_of_channels}, L={layers} | "
                f"avg={int(avg):,} cycles | "
                f"min={int(mn):,} | "
                f"max={int(mx):,}\n"
            )
    finally:
        f_main.close()
        f_run.close()

    print(f"\n\033[36m[INFO]\033[0m Saved {IMPLEMENTATION_NAME} cycle results to {OUTPUT_FILE}")
    print(f"\033[36m[INFO]\033[0m Per-run snapshot:               {RUN_FILE}")

    print("\n\033[32m========================================\033[0m")
    print("\033[32m  C++ cycle benchmark completed\033[0m")
    print("\033[32m========================================\033[0m")


if __name__ == "__main__":
    benchmark_cycles()
