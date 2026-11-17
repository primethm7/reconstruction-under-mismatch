"""Numerical checks of the blur operator conventions."""

import numpy as np
import pytest
import torch

from reconstruction_mismatch.operators import (
    CircularConvolution,
    GaussianBlur,
    IdentityOperator,
    circular_convolve_spatial,
    gaussian_kernel,
    kernel_to_psf,
)

DT = torch.float64


@pytest.mark.parametrize("sigma", [0.5, 1.0, 1.5, 2.25, 3.0])
def test_gaussian_kernel_normalised_symmetric(sigma):
    k = gaussian_kernel(sigma)
    assert k.shape[0] % 2 == 1 and k.shape == (k.shape[0], k.shape[0])
    assert abs(k.sum() - 1.0) < 1e-12
    assert np.allclose(k, k.T) and np.allclose(k, k[::-1, ::-1])
    r = k.shape[0] // 2
    assert k[r, r] == k.max()  # centre is the peak


def test_zero_sigma_is_identity_kernel():
    assert gaussian_kernel(0.0).shape == (1, 1)


def test_psf_centre_at_origin():
    k = np.zeros((5, 5))
    k[2, 2] = 1.0
    psf = kernel_to_psf(k, (16, 16))
    assert psf[0, 0] == 1.0 and psf.sum() == 1.0


def test_identity_operator():
    x = torch.rand(3, 1, 16, 20, dtype=DT)
    op = IdentityOperator((16, 20), dtype=DT)
    assert torch.allclose(op(x), x, atol=1e-12)
    assert torch.allclose(op.adjoint(x), x, atol=1e-12)


def test_constant_image_is_preserved():
    op = GaussianBlur(1.5, (32, 32), dtype=DT)
    x = torch.full((2, 1, 32, 32), 0.37, dtype=DT)
    assert torch.allclose(op(x), x, atol=1e-12)


def test_mean_is_preserved():
    op = GaussianBlur(2.0, (32, 32), dtype=DT)
    x = torch.rand(4, 1, 32, 32, dtype=DT)
    assert torch.allclose(op(x).mean(dim=(-2, -1)), x.mean(dim=(-2, -1)), atol=1e-12)


def test_fft_matches_spatial_reference_gaussian():
    rng = np.random.default_rng(0)
    x = rng.random((2, 24, 24))
    k = gaussian_kernel(1.5)
    ref = circular_convolve_spatial(x, k)
    op = CircularConvolution(k, (24, 24), dtype=DT)
    out = op(torch.from_numpy(x).unsqueeze(1)).squeeze(1).numpy()
    assert np.max(np.abs(out - ref)) < 1e-12


def test_fft_matches_spatial_reference_asymmetric_kernel():
    """An asymmetric kernel catches flipped-kernel / off-by-one centring mistakes."""
    rng = np.random.default_rng(1)
    k = rng.random((5, 7))
    k /= k.sum()
    x = rng.random((1, 16, 18))
    ref = circular_convolve_spatial(x, k)
    op = CircularConvolution(k, (16, 18), dtype=DT)
    out = op(torch.from_numpy(x).unsqueeze(1)).squeeze(1).numpy()
    assert np.max(np.abs(out - ref)) < 1e-12


def test_impulse_response_is_centred_kernel():
    k = gaussian_kernel(1.0)
    r = k.shape[0] // 2
    op = CircularConvolution(k, (16, 16), dtype=DT)
    delta = torch.zeros(1, 1, 16, 16, dtype=DT)
    delta[..., 8, 8] = 1.0
    out = op(delta)[0, 0].numpy()
    assert np.allclose(out[8 - r : 8 + r + 1, 8 - r : 8 + r + 1], k, atol=1e-12)


@pytest.mark.parametrize("kernel_kind", ["gauss", "asym"])
def test_adjoint_inner_product(kernel_kind):
    rng = np.random.default_rng(2)
    k = gaussian_kernel(1.7) if kernel_kind == "gauss" else rng.random((3, 5))
    op = CircularConvolution(k, (20, 28), dtype=DT)
    x = torch.from_numpy(rng.standard_normal((3, 1, 20, 28)))
    z = torch.from_numpy(rng.standard_normal((3, 1, 20, 28)))
    lhs = (op(x) * z).sum()
    rhs = (x * op.adjoint(z)).sum()
    assert abs(float(lhs - rhs)) / abs(float(lhs)) < 1e-12


def test_adjoint_is_matrix_transpose():
    rng = np.random.default_rng(3)
    k = rng.random((3, 3))
    op = CircularConvolution(k, (6, 6), dtype=DT)
    a = op.to_dense()
    z = rng.standard_normal((6, 6))
    adj = op.adjoint(torch.from_numpy(z).view(1, 1, 6, 6)).numpy().reshape(-1)
    assert np.allclose(adj, a.T @ z.reshape(-1), atol=1e-12)


def test_float32_and_float64_agree():
    x = torch.rand(2, 1, 32, 32, dtype=DT)
    a64 = GaussianBlur(1.5, (32, 32), dtype=torch.float64)(x)
    a32 = GaussianBlur(1.5, (32, 32), dtype=torch.float32)(x.float())
    assert torch.allclose(a32.double(), a64, atol=1e-6)


def test_shape_mismatch_raises():
    op = GaussianBlur(1.0, (16, 16))
    with pytest.raises(ValueError):
        op(torch.zeros(1, 1, 16, 17))
