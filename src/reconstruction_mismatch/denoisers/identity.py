"""Identity 'denoiser' D(x, sigma) = x, used to test PnP behaviour in a controlled way."""

from __future__ import annotations

import torch


class IdentityDenoiser:
    name = "identity"

    def __call__(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        return x
