"""Compute-relevant reconstruction code (hashed into the run cache key).

`reconstruct_and_score` receives observation-side inputs only (y, a reconstructor and the
assumed sigma); it has no access to ground truth or to the true operator.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from ..metrics.scores import operator_sensitivity, relative_residual, residual_autocorrelation
from ..operators import GaussianBlur
from ..reconstruction import PnPADMM, TikhonovReconstructor


def build_reconstructor(method: str, sel: dict, denoiser, cfg: dict):
    if method == "tikhonov":
        return TikhonovReconstructor(sel["tikhonov"]["lam"], clip=cfg["tikhonov"]["clip_output"])
    if method == "pnp_admm":
        p = sel["pnp_admm"]
        return PnPADMM(denoiser, p["rho"], p["sigma_d"], p["n_iter"], clip=cfg["pnp"]["clip_output"],
                       divergence_bound=cfg["pnp"]["divergence_bound"], denoiser_name=p["denoiser"]["denoiser"])
    raise ValueError(method)


def reconstruct_and_score(
    y: torch.Tensor, rec, sigma_assumed: float, truncate: float, score_cfg: dict, batch_size: int
) -> dict[str, np.ndarray]:
    """Reconstruct every observation with the assumed operator and compute label-free scores.

    Inputs are observation-side only: y, the reconstructor, sigma_assumed. No ground truth.
    """
    shape = tuple(y.shape[-2:])
    dev = y.device
    op = GaussianBlur(sigma_assumed, shape, truncate=truncate, device=dev)

    def rec_with_sigma(yb: torch.Tensor, s: float) -> torch.Tensor:
        return rec.reconstruct(yb, GaussianBlur(s, shape, truncate=truncate, device=dev))[0]

    keys = ["x_hat", "x_raw", "rel_residual", "residual_acf", "sensitivity", "runtime_s", "sens_runtime_s",
            "iterations", "diverged", "final_rel_change"]
    acc: dict[str, list] = {k: [] for k in keys}
    for yb in torch.split(y, batch_size):
        b = yb.shape[0]
        x, info = rec.reconstruct(yb, op)
        t0 = time.perf_counter()
        sens, _ = operator_sensitivity(yb, x, rec_with_sigma, sigma_assumed, delta=score_cfg["sensitivity_delta"],
                                       eps=score_cfg["residual_eps"])
        acc["sens_runtime_s"].append(np.full(b, (time.perf_counter() - t0) / b))
        acc["x_hat"].append(x.squeeze(1).cpu().numpy().astype(np.float32))
        acc["x_raw"].append(info["raw"].squeeze(1).cpu().numpy().astype(np.float32))
        acc["rel_residual"].append(relative_residual(yb, x, op, eps=score_cfg["residual_eps"]).cpu().numpy())
        acc["residual_acf"].append(residual_autocorrelation(yb, x, op, max_lag=int(score_cfg["acf_max_lag"])).cpu().numpy())
        acc["sensitivity"].append(sens.cpu().numpy())
        acc["runtime_s"].append(np.full(b, info["batch_seconds"] / b))
        acc["iterations"].append(np.asarray(info["iterations"]))
        acc["diverged"].append(np.asarray(info["diverged"]))
        acc["final_rel_change"].append(np.asarray(info["final_rel_change"], dtype=np.float64))
    return {k: np.concatenate(v) for k, v in acc.items()}
