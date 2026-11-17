"""Tikhonov and PnP-ADMM correctness checks."""

import numpy as np
import pytest
import torch

from reconstruction_mismatch.denoisers import IdentityDenoiser
from reconstruction_mismatch.operators import CircularConvolution, GaussianBlur, IdentityOperator, circular_convolve_spatial, gaussian_kernel
from reconstruction_mismatch.reconstruction import (
    PnPADMM,
    TikhonovReconstructor,
    pnp_data_step,
    tikhonov_dense,
    tikhonov_fft,
    tikhonov_gradient,
    tikhonov_gradient_descent,
    tikhonov_objective,
)

DT = torch.float64


def _spatial_dense_matrix(kernel, shape):
    """Dense A built from the spatial reference implementation (no FFT involved)."""
    n = shape[0] * shape[1]
    cols = []
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0
        cols.append(circular_convolve_spatial(e.reshape(shape), kernel).reshape(-1))
    return np.stack(cols, axis=1)


def test_tikhonov_identity_operator_closed_form():
    y = torch.rand(2, 1, 8, 8, dtype=DT)
    lam = 0.3
    z = tikhonov_fft(y, IdentityOperator((8, 8), dtype=DT), lam)
    assert torch.allclose(z, y / (1 + lam), atol=1e-12)


@pytest.mark.parametrize("lam", [1e-4, 1e-2, 0.5])
def test_tikhonov_fft_matches_dense_direct_solve(lam):
    rng = np.random.default_rng(0)
    shape = (14, 14)
    k = gaussian_kernel(1.5)
    a = _spatial_dense_matrix(k, shape)
    y = rng.random(shape)
    z_dense = tikhonov_dense(y, a, lam)
    op = CircularConvolution(k, shape, dtype=DT)
    z_fft = tikhonov_fft(torch.from_numpy(y).view(1, 1, *shape), op, lam)[0, 0].numpy()
    assert np.max(np.abs(z_fft - z_dense)) < 1e-8 * max(1.0, np.max(np.abs(z_dense)))


def test_tikhonov_fft_minimises_objective_and_gradient_zero():
    torch.manual_seed(0)
    op = GaussianBlur(1.5, (32, 32), dtype=DT)
    y = torch.rand(2, 1, 32, 32, dtype=DT)
    lam = 1e-2
    z = tikhonov_fft(y, op, lam)
    g = tikhonov_gradient(z, y, op, lam)
    assert g.abs().max() < 1e-10
    f0 = tikhonov_objective(z, y, op, lam)
    for _ in range(5):
        f1 = tikhonov_objective(z + 1e-3 * torch.randn_like(z), y, op, lam)
        assert torch.all(f1 > f0)


def test_gradient_matches_autograd_and_finite_difference():
    torch.manual_seed(1)
    op = GaussianBlur(1.2, (16, 16), dtype=DT)
    y = torch.rand(1, 1, 16, 16, dtype=DT)
    lam = 0.05
    z = torch.rand(1, 1, 16, 16, dtype=DT, requires_grad=True)
    f = tikhonov_objective(z, y, op, lam).sum()
    (g_auto,) = torch.autograd.grad(f, z)
    g_analytic = tikhonov_gradient(z.detach(), y, op, lam)
    assert torch.allclose(g_auto, g_analytic, atol=1e-12)
    # central finite differences on a few random coordinates
    zd = z.detach()
    h = 1e-6
    for idx in [(0, 0, 0, 0), (0, 0, 5, 7), (0, 0, 15, 3)]:
        e = torch.zeros_like(zd)
        e[idx] = h
        fd = (tikhonov_objective(zd + e, y, op, lam) - tikhonov_objective(zd - e, y, op, lam)).sum() / (2 * h)
        assert abs(float(fd) - float(g_analytic[idx])) < 1e-6


def test_objective_normalisation_constants():
    """J(z) = 0.5||Az-y||^2 + lam/2 ||z||^2 with plain sums (no 1/N averaging)."""
    op = IdentityOperator((4, 4), dtype=DT)
    z = torch.ones(1, 1, 4, 4, dtype=DT)
    y = torch.zeros(1, 1, 4, 4, dtype=DT)
    # ||z - y||^2 = 16, ||z||^2 = 16  ->  0.5*16 + 0.5*lam*16
    assert float(tikhonov_objective(z, y, op, 0.25)) == pytest.approx(8.0 + 2.0)


def test_gradient_descent_agrees_with_closed_form():
    op = GaussianBlur(1.0, (16, 16), dtype=DT)
    y = torch.rand(1, 1, 16, 16, dtype=DT)
    lam = 0.1
    z_cf = tikhonov_fft(y, op, lam)
    z_gd = tikhonov_gradient_descent(y, op, lam, n_iter=400)
    assert torch.allclose(z_gd, z_cf, atol=1e-8)


def test_tikhonov_reconstructor_clipping_is_separate():
    op = GaussianBlur(1.5, (16, 16), dtype=DT)
    y = torch.rand(1, 1, 16, 16, dtype=DT) * 2 - 0.5
    rec = TikhonovReconstructor(1e-3, clip=True)
    x, info = rec.reconstruct(y, op)
    assert torch.allclose(info["raw"], tikhonov_fft(y, op, 1e-3))
    assert x.min() >= 0 and x.max() <= 1
    assert torch.allclose(x, info["raw"].clamp(0, 1))


# --- PnP-ADMM ---------------------------------------------------------------------------


@pytest.mark.parametrize("rho", [0.05, 1.0, 10.0])
def test_pnp_data_step_matches_dense_solve(rho):
    rng = np.random.default_rng(4)
    shape = (14, 14)
    k = gaussian_kernel(1.3)
    a = _spatial_dense_matrix(k, shape)
    y = rng.random(shape)
    w = rng.random(shape)
    lhs = a.T @ a + rho * np.eye(a.shape[1])
    rhs = a.T @ y.reshape(-1) + rho * w.reshape(-1)
    x_dense = np.linalg.solve(lhs, rhs).reshape(shape)
    op = CircularConvolution(k, shape, dtype=DT)
    x = pnp_data_step(torch.from_numpy(y).view(1, 1, *shape), torch.from_numpy(w).view(1, 1, *shape), op, rho)
    assert np.max(np.abs(x[0, 0].numpy() - x_dense)) < 1e-9


def test_pnp_data_step_optimality_condition():
    op = GaussianBlur(1.5, (32, 32), dtype=DT)
    y = torch.rand(2, 1, 32, 32, dtype=DT)
    w = torch.rand(2, 1, 32, 32, dtype=DT)
    rho = 0.3
    x = pnp_data_step(y, w, op, rho)
    grad = op.adjoint(op(x) - y) + rho * (x - w)
    assert grad.abs().max() < 1e-10


def test_pnp_identity_denoiser_identity_operator_closed_form():
    """A = I, D = I: u stays 0 and x_k = y + (rho/(1+rho))^k (x_0 - y) with x_0 = y => x_k = y.
    Use a non-trivial start by checking the recursion on v directly."""
    torch.manual_seed(0)
    y = torch.rand(2, 1, 8, 8, dtype=DT)
    op = IdentityOperator((8, 8), dtype=DT)
    pnp = PnPADMM(IdentityDenoiser(), rho=0.7, sigma_d=0.0, n_iter=5, clip=False)
    x, info = pnp.run(y, op)
    assert torch.allclose(x, y, atol=1e-12)  # fixed point: x0 = y already optimal
    assert not info["diverged"].any()


def test_pnp_identity_denoiser_recursion_from_offset_start():
    """With D = I the iteration is a proximal-point method on 0.5||Ax-y||^2.
    For A = I and v0 = y + c: v_k - y = (rho/(1+rho))^k c."""
    y = torch.rand(1, 1, 8, 8, dtype=DT)
    c = torch.rand(1, 1, 8, 8, dtype=DT)
    rho = 0.5
    op = IdentityOperator((8, 8), dtype=DT)
    v = y + c
    u = torch.zeros_like(y)
    for k in range(1, 6):
        x = pnp_data_step(y, v - u, op, rho)
        v_new = IdentityDenoiser()(x + u, 0.0)
        u = u + x - v_new
        v = v_new
        assert torch.allclose(u, torch.zeros_like(u), atol=1e-14)
        assert torch.allclose(v - y, (rho / (1 + rho)) ** k * c, atol=1e-12)


def test_pnp_identity_denoiser_converges_to_inverse_filter():
    """With D = I and invertible A (well-conditioned blur), PnP-ADMM -> A^{-1} y."""
    op = GaussianBlur(0.6, (16, 16), dtype=DT)
    x_true = torch.rand(1, 1, 16, 16, dtype=DT)
    y = op(x_true)
    pnp = PnPADMM(IdentityDenoiser(), rho=0.05, sigma_d=0.0, n_iter=400, clip=False)
    x, _ = pnp.run(y, op)
    assert torch.allclose(x, x_true, atol=1e-6)


def test_pnp_divergence_detection():
    class Exploding:
        def __call__(self, x, sigma):
            return x * 1e6

    op = GaussianBlur(1.0, (12, 12), dtype=DT)
    y = torch.rand(3, 1, 12, 12, dtype=DT)
    pnp = PnPADMM(Exploding(), rho=1.0, sigma_d=0.02, n_iter=10, clip=False)
    x, info = pnp.run(y, op)
    assert info["diverged"].all()
    assert torch.isfinite(x).all()


def test_pnp_nan_detection():
    class NaNDenoiser:
        def __call__(self, x, sigma):
            out = x.clone()
            out[0] = float("nan")
            return out

    op = GaussianBlur(1.0, (12, 12), dtype=DT)
    y = torch.rand(2, 1, 12, 12, dtype=DT)
    _, info = PnPADMM(NaNDenoiser(), rho=1.0, sigma_d=0.02, n_iter=3).run(y, op)
    assert info["diverged"].tolist() == [True, False]
    assert info["diverged_at"][0] == 1
