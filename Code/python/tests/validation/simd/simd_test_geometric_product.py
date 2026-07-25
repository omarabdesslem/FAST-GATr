import csv
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch


VARIANT = "simd"
TEST_NAME = "geometric_product_validation"
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

from ezgatr.nn.functional import geometric_product as geometric_product_pytorch
from ops.linear import geometric_product as geometric_product_asl


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


def test_geometric_product():
    print("\033[35m========================================\033[0m")
    print("\033[35m  EzGATr geometric_product Validation Test (SIMD)\033[0m")
    print("\033[35m========================================\033[0m\n")

    test_cases = [
        # (batch, tokens, channels)
        (1, 4,   2),
        (1, 16,  2),
        (2, 32,  16),
        (8, 64,  16),
        (8, 128, 16),
        (8, 256, 16),
        (8, 256, 32),
    ]

    all_passed = True
    rows = []

    print("\033[36m[INFO]\033[0m Comparing Python geometric_product with vectorized ASL/C++ geometric_product\n")

    for batch, tokens, channels in test_cases:
        torch.manual_seed(42)

        x = torch.randn(batch, tokens, channels, 16)
        y = torch.randn(batch, tokens, channels, 16)

        with torch.no_grad():
            expected = geometric_product_pytorch(x, y)
            result = geometric_product_asl(x, y)

        max_error = (expected - result).abs().max().item()
        passes = torch.allclose(expected, result, atol=1e-5, rtol=1e-5)
        rows.append(
            {
                "batch": batch,
                "tokens": tokens,
                "channels": channels,
                "max_error": f"{max_error:.8e}",
                "atol": 1e-5,
                "rtol": 1e-5,
                "passed": passes,
            }
        )

        if passes:
            print(
                f"\033[32m[Passed]\033[0m "
                f"(B={batch}, T={tokens}, C={channels}) "
                f"max_err={max_error:.2e}"
            )
        else:
            all_passed = False
            print(
                f"\033[31m[Failed]\033[0m "
                f"(B={batch}, T={tokens}, C={channels}) "
                f"max_err={max_error:.2e}"
            )

    if all_passed:
        print("\n\033[32m========================================\033[0m")
        print("\033[32m  geometric_product validation completed successfully\033[0m")
        print("\033[32m========================================\033[0m")
    else:
        print("\n\033[31m========================================\033[0m")
        print("\033[31m  geometric_product validation failed\033[0m")
        print("\033[31m========================================\033[0m")

    output_file = write_results_csv(rows)
    print(f"\n\033[36m[INFO]\033[0m Saved validation results to {output_file}")


if __name__ == "__main__":
    test_geometric_product()
