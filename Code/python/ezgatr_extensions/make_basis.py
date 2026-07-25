from __future__ import annotations

from pathlib import Path

import torch
from ezgatr.nn.functional.linear import _load_bilinear_basis


ROOT = Path(__file__).resolve().parents[1]
BASIS_DIR = ROOT / "basis"


class BasisModule(torch.nn.Module):
    def __init__(self, b: torch.Tensor):
        super().__init__()
        self.b = b


def main() -> None:
    BASIS_DIR.mkdir(exist_ok=True)

    device = torch.device("cpu")
    dtype = torch.float32

    for kind, filename in [
        ("gp", "geometric_product_cpp.pt"),
        ("op", "outer_product_cpp.pt"),
    ]:
        basis = _load_bilinear_basis(kind, device, dtype).contiguous()
        assert basis.shape == (16, 16, 16), f"unexpected shape {basis.shape}"

        module = torch.jit.script(BasisModule(basis))
        out_path = BASIS_DIR / filename
        torch.jit.save(module, str(out_path))

        print(
            f"wrote {out_path} "
            f"shape={tuple(basis.shape)} "
            f"dtype={basis.dtype}"
        )


if __name__ == "__main__":
    main()