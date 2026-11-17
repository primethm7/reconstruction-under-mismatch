"""Label-free ("ground-truth-free") failure scores.

None of these functions accepts the ground truth or the true operator; they see only the
observation y, the reconstruction x_hat, the *assumed* operator, and (for sensitivity) the
reconstruction procedure itself. Higher score = more suspicious, for all three (the
direction is fixed before looking at test data).

1. relative_residual = ||A_assumed x_hat - y||_2 / (||y||_2 + eps)

2. residual_autocorrelation: r = A_assumed x_hat - y, r_c = r - mean(r).
   R(d) = sum_n r_c[n] r_c[n + d]   (circular autocorrelation, computed by FFT)
   score = mean_{d in W} |R(d)| / R(0),  W = {(dy, dx): |dy|, |dx| <= L} \\ {(0, 0)}
   With L = 3 this averages 48 lags. A white residual gives values near 1/sqrt(HW) per lag;
   spatially structured residual (edges left unexplained) gives larger values. Range [0, 1].

3. operator_sensitivity: re-run the SAME reconstructor on the SAME y with
   sigma_+ = sigma_assumed (1 + delta) and sigma_- = sigma_assumed (1 - delta):
   score = ||x_+ - x_-||_2 / (2 delta (||x_hat||_2 + eps))
   delta is a *relative* perturbation (dimensionless); the actual perturbation is
   +/- delta * sigma_assumed pixels. The score is the relative image change per unit
   relative change of the assumed blur width (a finite-difference elasticity).
   It is a sensitivity, not a posterior or calibrated uncertainty.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from ..operators import CircularConvolution


def _norm(t: torch.Tensor) -> torch.Tensor:
    return t.flatten(1).norm(dim=1)


def relative_residual(y: torch.Tensor, x_hat: torch.Tensor, op_assumed: CircularConvolution, eps: float = 1e-12) -> torch.Tensor:
    return _norm(op_assumed(x_hat) - y) / (_norm(y) + eps)


def residual_autocorrelation(
    y: torch.Tensor, x_hat: torch.Tensor, op_assumed: CircularConvolution, max_lag: int = 3, eps: float = 1e-20
) -> torch.Tensor:
    r = op_assumed(x_hat) - y
    r = r - r.mean(dim=(-2, -1), keepdim=True)
    spec = torch.fft.fft2(r)
    acf = torch.fft.ifft2(spec.real**2 + spec.imag**2).real  # R(d) at index d mod (H, W)
    h, w = acf.shape[-2:]
    vals = []
    for dy in range(-max_lag, max_lag + 1):
        for dx in range(-max_lag, max_lag + 1):
            if dy == 0 and dx == 0:
                continue
            vals.append(acf[..., dy % h, dx % w].abs())
    lagged = torch.stack(vals, dim=-1).mean(dim=-1)
    return (lagged / (acf[..., 0, 0] + eps)).flatten()


def operator_sensitivity(
    y: torch.Tensor,
    x_hat: torch.Tensor,
    reconstruct_with_sigma: Callable[[torch.Tensor, float], torch.Tensor],
    sigma_assumed: float,
    delta: float = 0.05,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, dict]:
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    s_plus = sigma_assumed * (1 + delta)
    s_minus = sigma_assumed * (1 - delta)
    x_plus = reconstruct_with_sigma(y, s_plus)
    x_minus = reconstruct_with_sigma(y, s_minus)
    score = _norm(x_plus - x_minus) / (2 * delta * (_norm(x_hat) + eps))
    return score, {"sigma_plus": s_plus, "sigma_minus": s_minus}
