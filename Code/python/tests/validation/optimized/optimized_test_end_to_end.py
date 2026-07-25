import csv
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch


VARIANT = "optimized"
TEST_NAME = "end_to_end_validation"
MAPE_DENOMINATOR_EPS = 1e-6
MAPE_THRESHOLD_PERCENT = 0.1
CODE_DIR = Path(__file__).resolve().parents[3]
CPP_IMPL = VARIANT


def build_cpp_impl() -> None:
    ops_dir = CODE_DIR / "ops"
    if ops_dir.exists():
        for pattern in ("*.so", "*.pyd", "*.dll"):
            for path in ops_dir.glob(pattern):
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

from ezgatr.nets.mv_only_gatr import MVOnlyGATrConfig, MVOnlyGATrModel
from ezgatr_extensions.asl_mv_only_gatr import ASLMVOnlyGATrModel


def mean_absolute_percentage_error(expected: torch.Tensor, result: torch.Tensor) -> float:
    denominator = expected.abs().clamp_min(MAPE_DENOMINATOR_EPS)
    return ((expected - result).abs() / denominator).mean().item() * 100.0


def validation_output_file() -> Path:
    date = datetime.now().strftime("%Y-%m-%d")
    run_id = os.environ.get("VALIDATION_RUN_ID", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    team80_dir = Path(__file__).resolve().parents[5]
    results_dir = team80_dir / "Results" / "validation" / VARIANT / date
    return results_dir / f"{TEST_NAME}_{run_id}.csv"


def write_results_csv(rows: list[dict]) -> Path:
    output_file = validation_output_file()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with output_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_file


def test_end_to_end():
    print("\033[35m========================================\033[0m")
    print("\033[35m  EzGATr End-to-End Validation Test\033[0m")
    print("\033[35m========================================\033[0m\n")

    test_cases = [
        # (batch, tokens, channels_in)
        (1, 4, 2),
        (1, 8, 2),
        (1, 16, 2),
        (2, 4, 2),
        (2, 16, 2),
        (4, 32, 2),
        (8, 64, 2),
        (8, 128, 2),
        (8, 256, 2),
        (1, 4, 1),
    ]

    all_passed = True
    rows = []

    print("\033[36m[INFO]\033[0m Comparing Python EzGATr with optimized ASL/C++ EzGATr")
    print("\033[36m[INFO]\033[0m If needed, run: python3 -m ezgatr_extensions.make_basis\n")

    for batch, tokens, ch_in in test_cases:
        torch.manual_seed(42)

        config = MVOnlyGATrConfig(size_channels_in=ch_in)
        net = MVOnlyGATrModel(config)

        asl_net = ASLMVOnlyGATrModel(config)
        asl_net.load_state_dict(net.state_dict(), strict=False)

        x = torch.randn(batch, tokens, ch_in, 16)

        with torch.no_grad():
            expected = net(x)
            result = asl_net(x)

        mape = mean_absolute_percentage_error(expected, result)
        passes = mape <= MAPE_THRESHOLD_PERCENT
        rows.append(
            {
                "batch": batch,
                "tokens": tokens,
                "channels": ch_in,
                "mape_percent": f"{mape:.8f}",
                "mape_threshold_percent": MAPE_THRESHOLD_PERCENT,
                "passed": passes,
            }
        )

        if passes:
            print(
                f"\033[32m[Passed]\033[0m "
                f"(B={batch}, T={tokens}, C={ch_in}) "
                f"MAPE={mape:.4f}%"
            )
        else:
            all_passed = False
            print(
                f"\033[31m[Failed]\033[0m "
                f"(B={batch}, T={tokens}, C={ch_in}) "
                f"MAPE={mape:.4f}%"
            )

    if all_passed:
        print("\n\033[32m========================================\033[0m")
        print("\033[32m  Validation completed successfully\033[0m")
        print("\033[32m========================================\033[0m")
    else:
        print("\n\033[31m========================================\033[0m")
        print("\033[31m  Validation failed\033[0m")
        print("\033[31m========================================\033[0m")

    output_file = write_results_csv(rows)
    print(f"\n\033[36m[INFO]\033[0m Saved validation results to {output_file}")


if __name__ == "__main__":
    test_end_to_end()
