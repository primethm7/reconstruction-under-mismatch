"""Reference-based image quality metrics (need ground truth -> evaluation stage only).

PSNR = 10 log10(data_range^2 / MSE) with data_range = 1.
SSIM = skimage.metrics.structural_similarity(data_range=1), default 7x7 uniform window.
Relative error = ||x_hat - x|| / ||x||.
"""

from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity


def mse(x_hat: np.ndarray, x_true: np.ndarray) -> float:
    return float(np.mean((np.asarray(x_hat, np.float64) - np.asarray(x_true, np.float64)) ** 2))


def psnr(x_hat: np.ndarray, x_true: np.ndarray, data_range: float = 1.0) -> float:
    m = mse(x_hat, x_true)
    if m == 0:
        return float("inf")
    return float(10.0 * np.log10(data_range**2 / m))


def ssim(x_hat: np.ndarray, x_true: np.ndarray, data_range: float = 1.0) -> float:
    return float(structural_similarity(np.asarray(x_true, np.float64), np.asarray(x_hat, np.float64), data_range=data_range))


def relative_error(x_hat: np.ndarray, x_true: np.ndarray) -> float:
    x_true = np.asarray(x_true, np.float64)
    return float(np.linalg.norm(np.asarray(x_hat, np.float64) - x_true) / (np.linalg.norm(x_true) + 1e-12))
