from __future__ import annotations

import argparse
import time

import torch

from ezgatr.nets.mv_only_gatr import MVOnlyGATrConfig, MVOnlyGATrModel
from ezgatr_extensions.asl_mv_only_gatr import ASLMVOnlyGATrModel


def parse_size(s: str) -> tuple[int, int, int]:
    try:
        b, t, c = map(int, s.split(","))
        return b, t, c
    except ValueError:
        raise argparse.ArgumentTypeError("Use format B,T,C, e.g. -s 1,4,2")


def run_model(name: str, model, x: torch.Tensor) -> torch.Tensor:
    print(f"\033[34m[STEP]\033[0m Running {name} model...", flush=True)

    start = time.perf_counter()
    with torch.no_grad():
        out = model(x)

    elapsed = time.perf_counter() - start
    print(
        f"\033[32m[OK]\033[0m {name} model finished in {elapsed:.3f}s "
        f"with output shape {tuple(out.shape)}",
        flush=True,
    )

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Runs a smoke test comparing Python EzGATr with ASL/C++ EzGATr."
    )
    parser.add_argument(
        "-s",
        "--size",
        type=parse_size,
        default=(1, 4, 2),
        help="Input size as batch,tokens,channels. Example: -s 1,4,2",
    )
    args = parser.parse_args()

    batch_size, tokens, channels = args.size
    device = "cpu"

    print("\033[35m========================================\033[0m")
    print("\033[35m  EzGATr Python vs ASL/C++ Smoke Test\033[0m")
    print("\033[35m========================================\033[0m\n")

    print(f"\033[36m[INFO]\033[0m Device: {device}", flush=True)
    print(
        f"\033[36m[INFO]\033[0m Input size: batch={batch_size}, "
        f"tokens={tokens}, channels={channels}, multivector_dim=16",
        flush=True,
    )

    print("\033[34m[STEP]\033[0m Creating config...", flush=True)
    config = MVOnlyGATrConfig(size_channels_in=channels)

    print("\033[34m[STEP]\033[0m Creating input tensor...", flush=True)
    x = torch.randn(batch_size, tokens, channels, 16).to(device)
    print(f"\033[32m[OK]\033[0m Input created with shape {tuple(x.shape)}", flush=True)

    print("\033[34m[STEP]\033[0m Creating baseline Python model...", flush=True)
    baseline_model = MVOnlyGATrModel(config).to(device)
    print("\033[32m[OK]\033[0m Baseline Python model created", flush=True)

    print("\033[34m[STEP]\033[0m Creating ASL/C++ model...", flush=True)
    asl_model = ASLMVOnlyGATrModel(config).to(device)
    print("\033[32m[OK]\033[0m ASL/C++ model created", flush=True)

    baseline_output = run_model("baseline Python", baseline_model, x)
    asl_output = run_model("ASL/C++", asl_model, x)

    print("\033[34m[STEP]\033[0m Comparing outputs...\n", flush=True)

    max_abs_diff = (baseline_output - asl_output).abs().max().item()

    print(f"\033[36m[INFO]\033[0m Baseline output shape: {tuple(baseline_output.shape)}")
    print(f"\033[36m[INFO]\033[0m ASL/C++ output shape:   {tuple(asl_output.shape)}")
    print(f"\033[36m[INFO]\033[0m Max absolute difference: {max_abs_diff}")
    print("\n\033[35m========================================\n  Test completed successfully\n========================================\033[0m")
if __name__ == "__main__":
    main()