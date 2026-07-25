from __future__ import annotations

import torch

from ezgatr.nn import EquiLinear
from ezgatr.nn.functional.linear import _compute_pin_equi_linear_basis

from ops import linear


class ASLEquiLinear(EquiLinear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer(
            "basis",
            _compute_pin_equi_linear_basis(
                torch.device("cpu"),
                torch.float32,
                self.normalize_basis,
            ),
        )
        self._packed_weight = None
        self._packed_weight_version = -1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            not torch.is_grad_enabled()
            and hasattr(linear, "pack_equi_linear_weight")
            and hasattr(linear, "equi_linear_packed")
        ):
            if self._packed_weight is None or self._packed_weight_version != self.weight._version:
                self._packed_weight = linear.pack_equi_linear_weight(self.weight)
                self._packed_weight_version = self.weight._version
            return linear.equi_linear_packed(x, self._packed_weight, self.basis, self.bias)
        return linear.equi_linear(x, self.weight, self.basis, self.bias)
