from __future__ import annotations

import torch
from ezgatr.nn.modules.norm import EquiRMSNorm

from ops import norm


class ASLEquiRMSNorm(EquiRMSNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return norm.equi_rms_norm(x, self.weight, self.eps)


def scaler_gated_gelu(x: torch.Tensor) -> torch.Tensor:
    return norm.scaler_gated_gelu(x)