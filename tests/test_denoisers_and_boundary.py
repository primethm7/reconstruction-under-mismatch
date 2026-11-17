"""Denoiser plumbing, training determinism, and the run-stage ground-truth boundary."""

from pathlib import Path

import numpy as np
import pytest
import torch

from reconstruction_mismatch.denoisers import SmallResidualDenoiser, load_cnn_denoiser
from reconstruction_mismatch.denoisers.ffdnet import FFDNetGray
from reconstruction_mismatch.denoisers.train import train_denoiser


def test_small_cnn_shapes_and_sigma_forms():
    m = SmallResidualDenoiser(depth=3, width=8)
    x = torch.rand(3, 1, 20, 24)
    assert m(x, 0.02).shape == x.shape
    assert m(x, torch.tensor([0.01, 0.02, 0.03])).shape == x.shape


def test_ffdnet_topology_and_odd_size():
    m = FFDNetGray()
    assert len([k for k in m.state_dict() if k.endswith("weight")]) == 15
    x = torch.rand(2, 1, 17, 22)
    assert m(x, 0.05).shape == x.shape


def test_kair_ffdnet_weights_if_downloaded():
    """Runs only after the optional extension downloaded the KAIR checkpoint."""
    from reconstruction_mismatch.utils.provenance import project_root

    path = project_root() / "artifacts" / "external" / "kair" / "ffdnet_gray.pth"
    if not path.exists():
        pytest.skip("KAIR FFDNet weights not downloaded (optional extension)")
    from reconstruction_mismatch.denoisers.ffdnet import load_kair_ffdnet
    from reconstruction_mismatch.data.splits import generate_split
    from reconstruction_mismatch.metrics.image import psnr

    den, _ = load_kair_ffdnet(path.parent)
    x = generate_split("test", 2, 77, 64)
    rng = np.random.default_rng(0)
    noisy = (x + 0.05 * rng.standard_normal(x.shape)).astype(np.float32)
    out = den(torch.from_numpy(noisy).unsqueeze(1), 0.05).squeeze(1).numpy()
    assert np.mean([psnr(out[i], x[i]) for i in range(2)]) > np.mean([psnr(noisy[i], x[i]) for i in range(2)]) + 2


def _tiny_train(tmp: Path):
    from reconstruction_mismatch.data.splits import generate_split

    train = generate_split("train", 6, 1, 64)
    held = generate_split("val", 2, 1, 64)
    cfg = {"iterations": 3, "batch_size": 4, "crop": 32, "lr": 1e-3, "lr_min": 1e-5, "sigma_range": [0.0, 0.1],
           "eval_every": 3, "eval_sigmas": [0.05], "seed": 3}
    meta = train_denoiser(train, held, {"depth": 3, "width": 8}, cfg, tmp, torch.device("cpu"), log=lambda *_: None)
    return meta


def test_training_is_deterministic_and_checkpoint_round_trips(tmp_path):
    m1 = _tiny_train(tmp_path / "a")
    m2 = _tiny_train(tmp_path / "b")
    assert m1["checkpoint_sha256"] == m2["checkpoint_sha256"]
    den, cfg = load_cnn_denoiser(tmp_path / "a" / "model.pt")
    assert cfg == {"arch": "SmallResidualDenoiser", "depth": 3, "width": 8}
    assert (tmp_path / "a" / "heldout.csv").exists()


def test_run_stage_never_reads_ground_truth(tmp_path, monkeypatch):
    """stage_run must work with ground-truth loaders and simulation metadata disabled."""
    from reconstruction_mismatch.config import load_config
    from reconstruction_mismatch.data.observations import save_observations, simulate_observations
    from reconstruction_mismatch.data.splits import generate_split
    from reconstruction_mismatch.denoisers import IdentityDenoiser
    from reconstruction_mismatch.experiments import pipeline as P

    imgs = generate_split("test", 2, 5, 64)
    obs, sim = simulate_observations(imgs, "test", 1.5, 0.02, [0], 5, 4.0)
    obs_dir = tmp_path / "obs"
    save_observations(obs, sim, obs_dir)

    def forbidden(*a, **k):
        raise AssertionError("ground truth / true operator accessed during the run stage")

    monkeypatch.setattr(P, "load_split_images", forbidden)
    monkeypatch.setattr(P, "load_simulation_meta", forbidden)
    cfg = load_config("smoke")
    cfg["operator"]["sigma_assumed"] = [1.5]
    ctx = P.Context(cfg, root=tmp_path, device=torch.device("cpu"), device_log={})
    ctx.log = __import__("logging").getLogger("test")
    (ctx.results).mkdir(parents=True)
    sel = {"tikhonov": {"lam": 1e-2}, "pnp_admm": {"rho": 0.5, "sigma_d": 0.02, "n_iter": 2,
                                                    "denoiser": {"denoiser": "identity"}}}
    run_dir = P.stage_run(ctx, obs_dir, sel, IdentityDenoiser())
    for m in ["tikhonov", "pnp_admm"]:
        d = np.load(run_dir / f"{m}__sa1.5000.npz")
        assert d["x_hat"].shape == (2, 64, 64)
        assert np.isfinite(d["rel_residual"]).all() and np.isfinite(d["sensitivity"]).all()
