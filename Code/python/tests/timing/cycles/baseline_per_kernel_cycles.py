"""
Uses cpp/_cycle_counter.hpp inside each kernel

Build the extensions with cycle counters enabled before running:
    set CYCLE_COUNT=1
    python setup.py build_ext --inplace

Run with:
    python -m tests.timing.cycles.baseline_per_kernel_cycles

Writes Results/timing_results/CSVs/<DATE>/c_per_kernel_cycles_results.csv with
one row per (kernel, B, T, C). Uses the same test cases as baseline_c_cycles.py.
"""

import csv
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
OUTPUT_FILE = RESULTS_DIR / "c_per_kernel_cycles_results.csv"
RUN_FILE = RESULTS_DIR / f"c_per_kernel_cycles_results_{DATE_TIME}.csv"
CPP_IMPL = "baseline"

WARMUP = 10
RUNS = 30
SEED = 42

# Shorter sweep than baseline_c_cycles.py because RDTSC scopes inside every
# kernel call add overhead.
TEST_CASES = [
    (1, 1, 1),
    (1, 4, 2),
    (8, 64, 2),
    (8, 256, 2),
    (8, 1024, 2),
]


def get_repeats(T):
    return RUNS


def build_cpp_impl():
    for pattern in ("*.so", "*.pyd", "*.dll"):
        for path in (CODE_DIR / "ops").glob(pattern):
            path.unlink()

    env = os.environ.copy()
    env["CPP_IMPL"] = CPP_IMPL
    env["CYCLE_COUNT"] = "1"
    subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace", "--force"],
        cwd=CODE_DIR,
        env=env,
        check=True,
    )


build_cpp_impl()

from ezgatr.nets.mv_only_gatr import MVOnlyGATrConfig
from ezgatr_extensions.asl_mv_only_gatr import ASLMVOnlyGATrModel
from ops import linear as ops_linear
from ops import dual as ops_dual
from ops import attention as ops_attention
from ops import norm as ops_norm


MODULES = [
    ("linear", ops_linear),
    ("dual", ops_dual),
    ("attention", ops_attention),
    ("norm", ops_norm),
]


def check_counters_enabled():
    for name, mod in MODULES:
        if not mod.cycle_counter_enabled():
            print(
                f"\033[31m[ERROR]\033[0m ops.{name} was built without CYCLE_COUNT.\n"
                "Rebuild with the env var set:\n"
                "  set CYCLE_COUNT=1   (cmd)  /  $env:CYCLE_COUNT='1' (powershell)\n"
                "  python setup.py build_ext --inplace",
                file=sys.stderr,
            )
            sys.exit(1)


def reset_all():
    for _, mod in MODULES:
        mod.cycle_counter_reset()


def collect_all():
    """Return list of (kernel_name, total_cycles, total_calls) across all modules."""
    out = []
    for _, mod in MODULES:
        for kname, (cycles, calls) in mod.cycle_counter_get().items():
            out.append((kname, cycles, calls))
    return out


def run_case(model, x, repeats):
    """Run WARMUP + repeats forward passes, return per-kernel totals (over repeats only)."""
    model.eval()

    with torch.no_grad():
        for _ in range(WARMUP):
            _ = model(x)

    reset_all()

    with torch.no_grad():
        for _ in range(repeats):
            _ = model(x)

    return collect_all()


def benchmark():
    print("\033[35m========================================\033[0m")
    print("\033[35m  EzGATr ASL Per-Kernel Cycle Benchmark\033[0m")
    print("\033[35m========================================\033[0m\n")

    check_counters_enabled()

    print(f"\033[36m[INFO]\033[0m Warmup runs: {WARMUP}")
    print(f"\033[36m[INFO]\033[0m C++ source folder: cpp/{CPP_IMPL}")
    print(f"\033[36m[INFO]\033[0m Timed repeats: adaptive (5/3 by T)")
    print(f"\033[36m[INFO]\033[0m Test cases: {len(TEST_CASES)}")
    print(f"\033[36m[INFO]\033[0m Output file: {OUTPUT_FILE}")
    print(f"\033[36m[INFO]\033[0m Per-run file: {RUN_FILE}\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    write_header = not OUTPUT_FILE.exists()
    f_main = OUTPUT_FILE.open("a", newline="")
    f_run = RUN_FILE.open("w", newline="")
    try:
        writer_main = csv.writer(f_main)
        writer_run = csv.writer(f_run)

        header = [
            "kernel",
            "batch",
            "tokens",
            "channels",
            "total_cycles",
            "total_calls",
            "avg_cycles_per_call",
            "avg_cycles_per_forward",
            "repeats",
        ]
        if write_header:
            writer_main.writerow(header)
        writer_run.writerow(header)

        for batch, tokens, channels in TEST_CASES:
            torch.manual_seed(SEED)

            print(
                f"\033[34m[STEP]\033[0m Running B={batch}, T={tokens}, C={channels}",
                flush=True,
            )

            config = MVOnlyGATrConfig(size_channels_in=channels)
            cpp_net = ASLMVOnlyGATrModel(config)

            x = torch.randn(batch, tokens, channels, 16)

            repeats = get_repeats(tokens)
            results = run_case(cpp_net, x, repeats)

            for kname, cycles, calls in results:
                avg_per_call = (cycles / calls) if calls else 0.0
                avg_per_forward = cycles / repeats

                row = [
                    kname,
                    batch,
                    tokens,
                    channels,
                    int(cycles),
                    int(calls),
                    int(avg_per_call),
                    int(avg_per_forward),
                    repeats,
                ]
                writer_main.writerow(row)
                writer_run.writerow(row)

            f_main.flush()
            f_run.flush()

            # Friendly summary line per case.
            total = sum(c for _, c, _ in results)
            print(f"\033[32m[CYCLES]\033[0m total measured: {int(total):,} cycles "
                  f"across {len(results)} kernels (over {repeats} forwards)")
            for kname, cycles, calls in sorted(results, key=lambda r: -r[1]):
                pct = (100.0 * cycles / total) if total else 0.0
                print(f"           {kname:<28} {int(cycles):>15,} ({pct:5.1f}%)  calls={calls}")
            print()
    finally:
        f_main.close()
        f_run.close()

    print(f"\033[36m[INFO]\033[0m Saved results to {OUTPUT_FILE}")
    print(f"\033[36m[INFO]\033[0m Per-run snapshot: {RUN_FILE}")
    print("\n\033[32m========================================\033[0m")
    print("\033[32m  Per-kernel cycle benchmark completed\033[0m")
    print("\033[32m========================================\033[0m")


if __name__ == "__main__":
    benchmark()
