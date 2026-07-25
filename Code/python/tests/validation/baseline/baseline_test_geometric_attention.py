import csv
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch


VARIANT = "baseline"
TEST_NAME = "geometric_attention_validation"
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

from ezgatr.nn.functional import equi_geometric_attention as py_attn
from ops.attention import equi_geometric_attention as asl_attn


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


def test_attention():
    print("\033[35m========================================\033[0m")
    print("\033[35m  EzGATr equi_geometric_attention Validation\033[0m")
    print("\033[35m========================================\033[0m\n")

    # (batch, heads, tokens, channels)
    test_cases = [
        (1, 4, 4,   32),
        (1, 4, 16,  32),
        (2, 4, 32,  32),
        (8, 4, 64,  32),
        (8, 4, 128, 32),
        (8, 4, 256, 32),
    ]

    # Match what MVOnlyGATrAttention passes in.
    kinds = {"ipa": None, "daa": None}

    all_passed = True
    rows = []

    print("\033[36m[INFO]\033[0m Comparing Python equi_geometric_attention with ASL/C++ version\n")

    for B, H, T, C in test_cases:
        torch.manual_seed(42)

        q = torch.randn(B, H, T, C, 16)
        k = torch.randn(B, H, T, C, 16)
        v = torch.randn(B, H, T, C, 16)

        # Match the model's attn_mix.exp() with default-init zeros: weights = 1.
        # Shape mirrors MVOnlyGATrAttention: (H, 1, C, 1).
        weight = [
            torch.zeros(H, 1, C, 1).exp(),  # ipa
            torch.zeros(H, 1, C, 1).exp(),  # daa
        ]

        with torch.no_grad():
            # File 1: returns bare tensor when no scalars present.
            expected_out = py_attn(
                q, k, v,
                kinds=kinds,
                weight=weight,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
                scale=None,
            )
            result_out = asl_attn(
                q, k, v,
                kinds=kinds,
                weight=weight,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
                scale=None,
            )

        expected, expected_scl = expected_out
        result, result_scl = result_out

        abs_err = (expected - result).abs()
        max_error = abs_err.max().item()
        mean_abs = abs_err.mean().item()
        
        passes = (
            abs_err.mean() < 1e-4 and
            abs_err.max() < 5e-2
        )
        rows.append(
            {
                "batch": B,
                "heads": H,
                "tokens": T,
                "channels": C,
                "max_error": f"{max_error:.8e}",
                "mean_abs_error": f"{mean_abs:.8e}",
                "mean_abs_threshold": 1e-4,
                "max_error_threshold": 5e-2,
                "passed": passes,
            }
        )

        if passes:
            print(
                f"\033[32m[Passed]\033[0m "
                f"(B={B}, H={H}, T={T}, C={C}) "
                f"max_err={max_error:.2e},"
                f"mean_err={mean_abs:.2e}"
            )
        else:
            all_passed = False
            print(
                f"\033[31m[Failed]\033[0m "
                f"(B={B}, H={H}, T={T}, C={C}) "
                f"max_err={max_error:.2e},"
                f"mean_err={mean_abs:.2e}"
            )

    if all_passed:
        print("\n\033[32m========================================\033[0m")
        print("\033[32m  Attention validation completed successfully\033[0m")
        print("\033[32m========================================\033[0m")
    else:
        print("\n\033[31m========================================\033[0m")
        print("\033[31m  Attention validation failed\033[0m")
        print("\033[31m========================================\033[0m")

    output_file = write_results_csv(rows)
    print(f"\n\033[36m[INFO]\033[0m Saved validation results to {output_file}")


if __name__ == "__main__":
    test_attention()
