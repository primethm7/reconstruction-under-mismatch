"""Tikhonov (ridge) deconvolution.

Objective (per image, sums over all pixels, no averaging):

    J(z) = 0.5 * ||A z - y||_2^2 + (lam / 2) * ||z||_2^2

Gradient: grad J(z) = A*(A z - y) + lam z.
Normal equations: (A*A + lam I) z = A* y. For a circular convolution A = F^-1 diag(H) F,
so the unique minimiser is

    z* = Re F^-1[ conj(H) * F y / (|H|^2 + lam) ]          (closed form, `tikhonov_fft`)

Clipping z* to [0, 1] is a *separate post-processing step* (TikhonovReconstructor with
clip=True); the closed-form solution itself is unconstrained.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from ..operators import CircularConvolution


def tikhonov_objective(z: torch.Tensor, y: torch.Tensor, op: CircularConvolution, lam: float) -> torch.Tensor:
    """Per-image objective value, shape = leading dims of z (e.g. (B, 1))."""
    r = op(z) - y
    return 0.5 * (r**2).sum(dim=(-2, -1)) + 0.5 * lam * (z**2).sum(dim=(-2, -1))


def tikhonov_gradient(z: torch.Tensor, y: torch.Tensor, op: CircularConvolution, lam: float) -> torch.Tensor:
    return op.adjoint(op(z) - y) + lam * z


def tikhonov_fft(y: torch.Tensor, op: CircularConvolution, lam: float) -> torch.Tensor:
    if lam < 0:
        raise ValueError("lam must be >= 0")
    num = torch.conj(op.otf) * torch.fft.fft2(y)
    den = op.otf_abs2 + lam
    return torch.fft.ifft2(num / den).real


def tikhonov_gradient_descent(
    y: torch.Tensor, op: CircularConvolution, lam: float, n_iter: int = 500, z0: torch.Tensor | None = None
) -> torch.Tensor:
    """Plain gradient descent with step 1/L, L = max|H|^2 + lam (verification only)."""
    lipschitz = float(op.otf_abs2.max()) + lam
    step = 1.0 / lipschitz
    z = torch.zeros_like(y) if z0 is None else z0.clone()
    for _ in range(n_iter):
        z = z - step * tikhonov_gradient(z, y, op, lam)
    return z


def tikhonov_dense(y: np.ndarray, a_matrix: np.ndarray, lam: float) -> np.ndarray:
    """Direct dense solve of (A^T A + lam I) z = A^T y (float64), independent of the FFT."""
    n = a_matrix.shape[1]
    lhs = a_matrix.T @ a_matrix + lam * np.eye(n)
    rhs = a_matrix.T @ y.reshape(-1)
    return np.linalg.solve(lhs, rhs).reshape(y.shape)


class TikhonovReconstructor:
    """Closed-form FFT Tikhonov; output optionally clipped to [0, 1] (post-processing)."""

    name = "tikhonov"

    def __init__(self, lam: float, clip: bool = True) -> None:
        self.lam = float(lam)
        self.clip = clip

    def reconstruct(self, y: torch.Tensor, op: CircularConvolution) -> tuple[torch.Tensor, dict]:
        t0 = time.perf_counter()
        z = tikhonov_fft(y, op, self.lam)
        x = z.clamp(0.0, 1.0) if self.clip else z
        elapsed = time.perf_counter() - t0
        finite = torch.isfinite(z).flatten(1).all(dim=1).cpu().numpy()
        info = {
            "raw": z,
            "iterations": np.zeros(y.shape[0], dtype=int),  # closed form: 0 iterations
            "diverged": ~finite,
            "final_rel_change": np.full(y.shape[0], np.nan),
            "batch_seconds": elapsed,
        }
        return x, info

    def describe(self) -> dict:
        return {"method": self.name, "lam": self.lam, "clip_output": self.clip}
