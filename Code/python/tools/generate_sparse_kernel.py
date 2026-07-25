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

def generate_kernel(name: str, basis: torch.Tensor) -> str:
    # collect non zero terms per outpub blade
    by_out: dict[int, list[tuple[int, int, int]]] = {k: [] for k in range(16)}
    for k in range(16):
        for i in range(16):
            for j in range(16):
                v = float(basis[k, i, j].item())
                if v == 0.0:
                    continue
                by_out[k].append((i, j, int(v)))

    L: list[str] = []
    L.append(f"static inline void {name}_kernel_one_batch(")
    L.append("    const float* __restrict x_row,")
    L.append("    const float* __restrict y_row,")
    L.append("    float*       __restrict o_row")
    L.append(") {")

    for v in range(16):
        L.append(f"    const float x{v} = x_row[{v}];")
    for v in range(16):
        L.append(f"    const float y{v} = y_row[{v}];")
    L.append("")

    for k in range(16):
        L.append(f"    float a{k} = 0.0f;")
    L.append("")

    for k in range(16):
        if not by_out[k]:
            continue
        for (i, j, s) in by_out[k]:
            op = "+=" if s == 1 else "-="
            L.append(f"    a{k} {op} x{i} * y{j};")
        L.append("")

    for k in range(16):
        L.append(f"    o_row[{k}] = a{k};")

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
    args.out.write_text(generate_kernel(name, basis))


if __name__ == "__main__":
    main()