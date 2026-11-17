"""Linear blur operators with circular (periodic) boundary conditions.

Conventions (used by *every* operator and solver in this package):

* Images are torch tensors of shape (B, 1, H, W) (any leading batch dims work).
* A kernel k of odd size (2r+1, 2r+1) is centred at index (r, r). It is embedded in an
  H x W array and circularly shifted so its centre sits at pixel (0, 0); its 2-D DFT is
  the optical transfer function (OTF) H.
* FFTs use torch's default ("backward") normalisation, so for real a, b
  ifft2(fft2(a) * fft2(b)) is exactly the circular convolution a (*) b.
* Forward:  A x  = Re ifft2( H        * fft2(x) )
  Adjoint:  A* z = Re ifft2( conj(H)  * fft2(z) )
  (A* is the exact adjoint w.r.t. the Euclidean inner product <a, b> = sum a*b.)
* Periodic boundaries are a modelling choice: they make A diagonal in the Fourier basis
  (exact adjoints, closed-form solves) but are not a physical model of real optics.
"""

from __future__ import annotations

import math

import numpy as np
import torch


def gaussian_kernel(sigma: float, truncate: float = 4.0) -> np.ndarray:
    """Normalised (sum = 1) isotropic Gaussian kernel, float64, odd size 2r+1.

    r = ceil(truncate * sigma). sigma <= 0 returns the 1x1 identity kernel.
    The truncated kernel is renormalised, so A preserves the mean intensity exactly.
    """
    if sigma <= 0:
        return np.ones((1, 1), dtype=np.float64)
    r = int(math.ceil(truncate * sigma))
    ax = np.arange(-r, r + 1, dtype=np.float64)
    g = np.exp(-(ax**2) / (2.0 * sigma**2))
    k = np.outer(g, g)
    return k / k.sum()


def kernel_to_psf(kernel: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Embed an odd-sized centred kernel into an array of `shape`, centre moved to (0, 0)."""
    kh, kw = kernel.shape
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError("kernel size must be odd so the centre pixel is well defined")
    if kh > shape[0] or kw > shape[1]:
        raise ValueError(f"kernel {kernel.shape} larger than image {shape}")
    psf = np.zeros(shape, dtype=np.float64)
    psf[:kh, :kw] = kernel
    return np.roll(psf, shift=(-(kh // 2), -(kw // 2)), axis=(0, 1))


def circular_convolve_spatial(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Reference implementation (no FFT): (k (*) x)[n] = sum_m k[m] x[n - m], periodic.

    Used only in tests to validate the FFT operator, including for asymmetric kernels.
    Works on arrays whose last two axes are spatial.
    """
    kh, kw = kernel.shape
    rh, rw = kh // 2, kw // 2
    out = np.zeros_like(x, dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            if kernel[i, j] == 0:
                continue
            out += kernel[i, j] * np.roll(x, shift=(i - rh, j - rw), axis=(-2, -1))
    return out


class CircularConvolution:
    """A x = k (*) x with periodic boundaries, evaluated with FFTs."""

    def __init__(
        self,
        kernel: np.ndarray,
        shape: tuple[int, int],
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.kernel = np.asarray(kernel, dtype=np.float64)
        self.shape = tuple(shape)
        self.device = torch.device(device)
        self.dtype = dtype
        psf = torch.from_numpy(kernel_to_psf(self.kernel, self.shape))
        otf = torch.fft.fft2(psf)  # complex128, computed in double precision
        cdtype = torch.complex64 if dtype == torch.float32 else torch.complex128
        self.otf = otf.to(device=self.device, dtype=cdtype)
        self.otf_abs2 = (self.otf.real**2 + self.otf.imag**2).to(dtype)

    # -- linear maps ---------------------------------------------------------------
    def _check(self, x: torch.Tensor) -> None:
        if tuple(x.shape[-2:]) != self.shape:
            raise ValueError(f"expected spatial shape {self.shape}, got {tuple(x.shape[-2:])}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check(x)
        return torch.fft.ifft2(self.otf * torch.fft.fft2(x)).real

    __call__ = forward

    def adjoint(self, z: torch.Tensor) -> torch.Tensor:
        self._check(z)
        return torch.fft.ifft2(torch.conj(self.otf) * torch.fft.fft2(z)).real

    def normal(self, x: torch.Tensor) -> torch.Tensor:
        """A* A x."""
        return torch.fft.ifft2(self.otf_abs2 * torch.fft.fft2(x)).real

    def to_dense(self) -> np.ndarray:
        """Dense matrix (N x N, N = H*W) built column-by-column. Small images only."""
        n = self.shape[0] * self.shape[1]
        if n > 4096:
            raise ValueError("to_dense is meant for tiny test problems")
        eye = torch.eye(n, dtype=self.dtype, device=self.device).reshape(n, 1, *self.shape)
        cols = self.forward(eye).reshape(n, n)  # row i = A e_i
        return cols.T.cpu().numpy().astype(np.float64)

    def describe(self) -> dict:
        return {"type": type(self).__name__, "kernel_shape": list(self.kernel.shape), "boundary": "circular"}


class GaussianBlur(CircularConvolution):
    def __init__(
        self,
        sigma: float,
        shape: tuple[int, int],
        truncate: float = 4.0,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.sigma = float(sigma)
        self.truncate = float(truncate)
        super().__init__(gaussian_kernel(sigma, truncate), shape, device=device, dtype=dtype)

    def describe(self) -> dict:
        d = super().describe()
        d.update({"sigma": self.sigma, "truncate": self.truncate})
        return d


class IdentityOperator(CircularConvolution):
    def __init__(self, shape, device="cpu", dtype=torch.float32) -> None:
        super().__init__(np.ones((1, 1)), shape, device=device, dtype=dtype)
