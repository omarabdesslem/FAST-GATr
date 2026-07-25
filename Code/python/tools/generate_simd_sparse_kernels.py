from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ezgatr.nn.functional.dual import _compute_efficient_join_kernel
from ezgatr.nn.functional.linear import _load_bilinear_basis

_CPU = torch.device("cpu")

def build_basis(op: str) -> torch.Tensor:
    if op == "gp":
        return _load_bilinear_basis("gp", _CPU, torch.float32)
    return _compute_efficient_join_kernel(_CPU, torch.float32)

def generate_simd_kernel(name: str, basis: torch.Tensor) -> str:
    # collect non zero terms per output blade
    by_out: dict[int, list[tuple[int, int, int]]] = {k: [] for k in range(16)}
    for k in range(16):
        for i in range(16):
            for j in range(16):
                v = float(basis[k, i, j].item())
                if v == 0.0:
                    continue
                by_out[k].append((i, j, int(v)))

    # ARM NEON
    L: list[str] = []
    L.append("#pragma once")
    L.append("#include <arm_neon.h>")
    L.append("")
    L.append(f"static inline void {name}_one_batch_simd("
             "const float32x4_t* x, const float32x4_t* y, float32x4_t* o) {")
    for k in range(16):
        L.append(f"    o[{k}] = vdupq_n_f32(0.0f);")
        for (i, j, s) in by_out[k]:
            fn = "vfmaq_f32" if s == 1 else "vfmsq_f32"
            L.append(f"    o[{k}] = {fn}(o[{k}], x[{i}], y[{j}]);")
        L.append("")
    L.append("}")
    L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", choices=["gp", "join"])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    name = args.name or args.op
    basis = build_basis(args.op)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(generate_simd_kernel(name, basis))


if __name__ == "__main__":
    main()
