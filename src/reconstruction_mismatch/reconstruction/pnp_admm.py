"""Plug-and-Play ADMM (Venkatakrishnan et al. 2013; Chan et al. 2017) for deblurring.

Splitting  min_x f(x) + g(v)  s.t. x = v,  with f(x) = 0.5 ||A x - y||^2 and g implicit.
Scaled-form ADMM with penalty rho > 0; the g-proximal step is replaced by a denoiser
D(., sigma_d):

    x^{k+1} = argmin_x 0.5 ||A x - y||^2 + (rho/2) ||x - (v^k - u^k)||^2
            = Re F^-1[ (conj(H) F y + rho F(v^k - u^k)) / (|H|^2 + rho) ]     (exact, FFT)
    v^{k+1} = D(x^{k+1} + u^k, sigma_d)
    u^{k+1} = u^k + x^{k+1} - v^{k+1}

Initialisation: x^0 = v^0 = y (the observation), u^0 = 0.
Stopping: a fixed number of iterations K (selected on validation data). The relative change
||v^K - v^{K-1}|| / ||v^{K-1}|| is recorded as a convergence diagnostic, not used to stop.
Output: v^K (the last denoiser output), clipped to [0, 1] as post-processing.

Notes (deliberately not over-claimed):
* With a learned CNN denoiser there is no explicit regulariser g for which this provably
  minimises an objective, and convergence is not guaranteed for the denoisers used here.
* Heuristically, if D were the proximal map of lam*g then sigma_d^2 ~ lam / rho; we treat
  (rho, sigma_d, K) as three hyper-parameters chosen on validation images.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable

import numpy as np
import torch

from ..operators import CircularConvolution

Denoiser = Callable[[torch.Tensor, float], torch.Tensor]


def pnp_data_step(y: torch.Tensor, w: torch.Tensor, op: CircularConvolution, rho: float) -> torch.Tensor:
    """argmin_x 0.5 ||A x - y||^2 + (rho/2) ||x - w||^2 (exact via FFT)."""
    num = torch.conj(op.otf) * torch.fft.fft2(y) + rho * torch.fft.fft2(w)
    return torch.fft.ifft2(num / (op.otf_abs2 + rho)).real


def _norm(t: torch.Tensor) -> torch.Tensor:
    return t.flatten(1).norm(dim=1)


class PnPADMM:
    name = "pnp_admm"

    def __init__(
        self,
        denoiser: Denoiser,
        rho: float,
        sigma_d: float,
        n_iter: int,
        clip: bool = True,
        divergence_bound: float = 1e3,
        denoiser_name: str = "unknown",
    ) -> None:
        if rho <= 0:
            raise ValueError("rho must be > 0")
        self.denoiser = denoiser
        self.rho = float(rho)
        self.sigma_d = float(sigma_d)
        self.n_iter = int(n_iter)
        self.clip = clip
        self.divergence_bound = float(divergence_bound)
        self.denoiser_name = denoiser_name

    @torch.no_grad()
    def run(
        self,
        y: torch.Tensor,
        op: CircularConvolution,
        record_at: Iterable[int] | None = None,
    ) -> tuple[torch.Tensor, dict]:
        t0 = time.perf_counter()
        record = set(int(k) for k in (record_at or []))
        b = y.shape[0]
        x = y.clone()
        v = y.clone()
        u = torch.zeros_like(y)
        diverged = torch.zeros(b, dtype=torch.bool, device=y.device)
        diverged_at = np.full(b, -1)
        snapshots: dict[int, torch.Tensor] = {}
        rel_change = torch.full((b,), float("nan"), device=y.device)
        for k in range(1, self.n_iter + 1):
            x_new = pnp_data_step(y, v - u, op, self.rho)
            v_new = self.denoiser(x_new + u, self.sigma_d)
            u_new = u + x_new - v_new
            bad = ~(
                torch.isfinite(x_new).flatten(1).all(1)
                & torch.isfinite(v_new).flatten(1).all(1)
                & torch.isfinite(u_new).flatten(1).all(1)
                & (v_new.flatten(1).abs().amax(1) < self.divergence_bound)
            )
            newly = bad & ~diverged
            for i in torch.nonzero(newly).flatten().tolist():
                diverged_at[i] = k
            diverged |= bad
            keep = (~diverged).view(-1, *([1] * (y.dim() - 1)))
            rel_change = torch.where(diverged, rel_change, _norm(v_new - v) / (_norm(v) + 1e-12))
            x = torch.where(keep, x_new, x)
            v = torch.where(keep, v_new, v)
            u = torch.where(keep, u_new, u)
            if k in record:
                snapshots[k] = (v.clamp(0, 1) if self.clip else v).clone()
        out = v.clamp(0.0, 1.0) if self.clip else v
        info = {
            "raw": v,
            "iterations": np.full(b, self.n_iter, dtype=int),
            "diverged": diverged.cpu().numpy(),
            "diverged_at": diverged_at,
            "final_rel_change": rel_change.cpu().numpy(),
            "snapshots": snapshots,
            "batch_seconds": time.perf_counter() - t0,
        }
        return out, info

    def reconstruct(self, y: torch.Tensor, op: CircularConvolution) -> tuple[torch.Tensor, dict]:
        return self.run(y, op)

    def describe(self) -> dict:
        return {
            "method": self.name,
            "rho": self.rho,
            "sigma_d": self.sigma_d,
            "n_iter": self.n_iter,
            "init": "x0 = v0 = y, u0 = 0",
            "output": "v^K clipped to [0,1]",
            "denoiser": self.denoiser_name,
            "clip_output": self.clip,
        }
