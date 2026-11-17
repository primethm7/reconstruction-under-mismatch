"""Image metrics, label-free scores (and their no-ground-truth API), failure-capture metrics."""

import inspect

import numpy as np
import pytest
import torch
from skimage.metrics import peak_signal_noise_ratio

from reconstruction_mismatch.metrics import failure as F
from reconstruction_mismatch.metrics import image as im
from reconstruction_mismatch.metrics import scores as S
from reconstruction_mismatch.operators import GaussianBlur
from reconstruction_mismatch.reconstruction import TikhonovReconstructor

DT = torch.float64

# ---- reference metrics ---------------------------------------------------------------


def test_psnr_known_value_and_skimage_agreement():
    x = np.zeros((10, 10))
    y = np.full((10, 10), 0.1)  # MSE = 0.01 -> 20 dB
    assert im.psnr(y, x) == pytest.approx(20.0)
    rng = np.random.default_rng(0)
    a = rng.random((32, 32))
    b = np.clip(a + 0.05 * rng.standard_normal((32, 32)), 0, 1)
    assert im.psnr(b, a) == pytest.approx(peak_signal_noise_ratio(a, b, data_range=1.0))


def test_psnr_identical_is_inf_ssim_identical_is_one():
    a = np.random.default_rng(1).random((32, 32))
    assert im.psnr(a, a) == float("inf")
    assert im.ssim(a, a) == pytest.approx(1.0)


def test_ssim_decreases_with_noise():
    rng = np.random.default_rng(2)
    a = rng.random((64, 64))
    s = [im.ssim(np.clip(a + n * rng.standard_normal(a.shape), 0, 1), a) for n in (0.01, 0.05, 0.2)]
    assert s[0] > s[1] > s[2]


def test_relative_error():
    a = np.ones((4, 4))
    assert im.relative_error(a * 1.1, a) == pytest.approx(0.1)


# ---- label-free scores ---------------------------------------------------------------

FORBIDDEN = {"x_true", "gt", "ground_truth", "target", "clean", "x", "sigma_true", "op_true", "mse", "psnr"}


@pytest.mark.parametrize("fn", [S.relative_residual, S.residual_autocorrelation, S.operator_sensitivity])
def test_score_api_has_no_ground_truth_argument(fn):
    params = set(inspect.signature(fn).parameters)
    assert not (params & FORBIDDEN), params & FORBIDDEN


def test_reconstruct_and_score_api_has_no_ground_truth_argument():
    from reconstruction_mismatch.experiments.pipeline import reconstruct_and_score

    params = set(inspect.signature(reconstruct_and_score).parameters)
    assert params == {"y", "rec", "sigma_assumed", "truncate", "score_cfg", "batch_size"}


def test_relative_residual_zero_for_exact_noiseless_fit():
    op = GaussianBlur(1.5, (32, 32), dtype=DT)
    x = torch.rand(2, 1, 32, 32, dtype=DT)
    y = op(x)
    assert torch.all(S.relative_residual(y, x, op) < 1e-12)


def test_residual_autocorrelation_white_vs_structured():
    torch.manual_seed(0)
    op = GaussianBlur(1.5, (64, 64), dtype=DT)
    x = torch.rand(1, 1, 64, 64, dtype=DT)
    y_white = op(x) + 0.02 * torch.randn(1, 1, 64, 64, dtype=DT)
    white = float(S.residual_autocorrelation(y_white, x, op)[0])
    # structured residual: smooth (blurred) error
    smooth_err = GaussianBlur(3.0, (64, 64), dtype=DT)(torch.randn(1, 1, 64, 64, dtype=DT))
    structured = float(S.residual_autocorrelation(op(x) + smooth_err, x, op)[0])
    assert white < 0.05 < 0.5 < structured <= 1.0


def test_sensitivity_zero_for_sigma_independent_reconstructor():
    y = torch.rand(2, 1, 16, 16, dtype=DT)
    score, info = S.operator_sensitivity(y, y, lambda yy, s: yy, sigma_assumed=1.5, delta=0.05)
    assert torch.all(score == 0)
    assert info["sigma_plus"] == pytest.approx(1.575) and info["sigma_minus"] == pytest.approx(1.425)


def test_sensitivity_matches_manual_finite_difference():
    torch.manual_seed(1)
    y = torch.rand(1, 1, 32, 32, dtype=DT)
    rec = TikhonovReconstructor(1e-2, clip=False)

    def f(yy, s):
        return rec.reconstruct(yy, GaussianBlur(s, (32, 32), dtype=DT))[0]

    x = f(y, 1.5)
    score, _ = S.operator_sensitivity(y, x, f, 1.5, delta=0.05)
    manual = (f(y, 1.575) - f(y, 1.425)).norm() / (2 * 0.05 * x.norm())
    assert float(score[0]) == pytest.approx(float(manual))
    assert float(score[0]) > 0


def test_sensitivity_rejects_bad_delta():
    with pytest.raises(ValueError):
        S.operator_sensitivity(torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4), lambda y, s: y, 1.0, delta=0.0)


# ---- failure-capture metrics ---------------------------------------------------------


def test_review_k_rounding():
    assert F.review_k(48, 0.1) == 5  # ceil(4.8)
    assert F.review_k(10, 0.3) == 3  # exact, no float spill-over to 4
    assert F.review_k(4, 0.1) == 1
    assert F.review_k(12, 1.0) == 12


def test_capture_perfect_reversed_random():
    labels = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], bool)
    perfect = np.arange(10, 0, -1, dtype=float)  # failures have the highest scores
    assert F.capture_rate(perfect, labels, 0.2) == 1.0
    assert F.capture_rate(-perfect, labels, 0.2) == 0.0
    assert F.capture_rate(perfect, labels, 0.2) == F.oracle_capture(labels, 0.2)
    assert F.random_capture(10, 0.2) == 0.2


def test_capture_ties_expected_value():
    labels = np.array([1, 0, 0, 0], bool)
    scores = np.zeros(4)  # all tied -> expected capture = k/n
    assert F.capture_rate(scores, labels, 0.5) == pytest.approx(0.5)
    scores = np.array([2.0, 1.0, 1.0, 1.0])
    labels = np.array([0, 1, 0, 0], bool)
    # top-2: item 0 for sure, then 1 of 3 tied items -> expected 1/3 of the failure
    assert F.capture_rate(scores, labels, 0.5) == pytest.approx(1 / 3)


def test_capture_undefined_without_failures():
    assert np.isnan(F.capture_rate(np.arange(5.0), np.zeros(5, bool), 0.2))
    assert np.isnan(F.oracle_capture(np.zeros(5, bool), 0.2))


def test_nan_scores_ranked_first():
    labels = np.array([0, 1, 0], bool)
    assert F.capture_rate(np.array([1.0, np.nan, 2.0]), labels, 0.34) == 1.0


def test_auroc_basic():
    assert F.auroc(np.array([0.9, 0.8, 0.1, 0.2]), np.array([1, 1, 0, 0], bool)) == 1.0
    assert F.auroc(np.array([0.5, 0.5, 0.5, 0.5]), np.array([1, 1, 0, 0], bool)) == 0.5
    assert np.isnan(F.auroc(np.array([1.0, 2.0]), np.array([0, 0], bool)))
