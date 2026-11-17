"""Synthetic data, split independence, observation simulation, reproducibility."""

import numpy as np
import pytest

from reconstruction_mismatch.data.observations import noise_rng, simulate_observations
from reconstruction_mismatch.data.splits import (
    build_synthetic_dataset,
    check_no_overlap,
    generate_split,
    image_rng,
)
from reconstruction_mismatch.data.synthetic import generate_phantom


def test_phantom_range_dtype_shape():
    x = generate_phantom(image_rng(1, "train", 0), size=128)
    assert x.shape == (128, 128) and x.dtype == np.float32
    assert x.min() >= 0.0 and x.max() <= 1.0
    assert x.std() > 0.05  # not a flat image


def test_phantom_reproducible_and_distinct():
    a = generate_phantom(image_rng(5, "test", 3))
    b = generate_phantom(image_rng(5, "test", 3))
    c = generate_phantom(image_rng(5, "test", 4))
    d = generate_phantom(image_rng(5, "val", 3))
    assert np.array_equal(a, b)
    assert not np.allclose(a, c) and not np.allclose(a, d)


def test_subset_property_quick_in_full():
    small = generate_split("test", 3, 11, 64)
    large = generate_split("test", 6, 11, 64)
    assert np.array_equal(small, large[:3])


def test_split_overlap_detection(tmp_path):
    cfg = {"n_train": 6, "n_val": 3, "n_test": 3}
    ddir, man = build_synthetic_dataset(cfg, 123, 64, tmp_path)
    check_no_overlap(man)  # does not raise
    hashes = [h for s in man["splits"].values() for h in s["image_sha256"]]
    assert len(hashes) == len(set(hashes)) == 12
    bad = {"splits": {"train": {"image_sha256": ["a", "b"]}, "test": {"image_sha256": ["b"]}}}
    with pytest.raises(RuntimeError):
        check_no_overlap(bad)


def test_dataset_cache_reuse_is_identical(tmp_path):
    cfg = {"n_train": 2, "n_val": 2, "n_test": 2}
    _, m1 = build_synthetic_dataset(cfg, 9, 64, tmp_path)
    _, m2 = build_synthetic_dataset(cfg, 9, 64, tmp_path)
    assert m1 == m2


def test_observation_model_and_noise_statistics():
    imgs = generate_split("test", 4, 2, 64)
    obs, sim = simulate_observations(imgs, "test", 1.5, 0.02, [0, 1], 2, 4.0)
    assert obs.y.shape == (8, 64, 64)
    assert list(obs.noise_seed) == [0] * 4 + [1] * 4
    assert sim["y_clipped"] is False
    # residual y - A x is the noise: check its std and that seeds give different noise
    from reconstruction_mismatch.operators import GaussianBlur
    import torch

    ax = GaussianBlur(1.5, (64, 64), dtype=torch.float64)(torch.from_numpy(imgs.astype(np.float64)).unsqueeze(1)).squeeze(1).numpy()
    eps0 = obs.y[:4] - ax
    eps1 = obs.y[4:] - ax
    assert abs(eps0.std() - 0.02) < 0.001
    assert abs(np.corrcoef(eps0.ravel(), eps1.ravel())[0, 1]) < 0.02


def test_observations_reproducible():
    imgs = generate_split("val", 2, 3, 64)
    o1, _ = simulate_observations(imgs, "val", 1.5, 0.02, [0], 3, 4.0)
    o2, _ = simulate_observations(imgs, "val", 1.5, 0.02, [0], 3, 4.0)
    assert np.array_equal(o1.y, o2.y)


def test_noise_streams_independent_of_split_and_index():
    a = noise_rng(1, "test", 0, 0).standard_normal(100)
    b = noise_rng(1, "val", 0, 0).standard_normal(100)
    c = noise_rng(1, "test", 1, 0).standard_normal(100)
    assert not np.allclose(a, b) and not np.allclose(a, c)
