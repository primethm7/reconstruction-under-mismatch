"""Training of the small residual CNN denoiser on the synthetic TRAIN split only.

* Inputs: random 64x64 crops of training phantoms, random flips / 90-degree rotations.
* Noise: x + sigma * n, n ~ N(0, I), sigma ~ U[sigma_min, sigma_max] per sample; the
  network receives sigma as a constant input map (non-blind, noise-conditional).
* Loss: mean squared error between D(x + sigma n, sigma) and x. Adam, cosine LR decay.
* Monitoring (no early stopping, no model selection): PSNR on held-out VALIDATION phantoms
  at fixed noise levels with a fixed noise realisation, before training (random init) and
  every `eval_every` iterations. The final iterate is the model that is used.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..metrics.image import psnr
from ..utils.io import save_json, sha256_file, sha256_obj
from .cnn import SmallResidualDenoiser


def denoiser_key(arch: dict, train_cfg: dict, train_sha: str, heldout_sha: str, code_sha: str) -> str:
    spec = {"arch": arch, "train": train_cfg, "train_sha": train_sha, "heldout_sha": heldout_sha, "code": code_sha}
    return "den-" + sha256_obj(spec)[:12]


def _heldout_noise(heldout: np.ndarray, sigmas: list[float], seed: int = 999) -> dict[float, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {s: (heldout + s * rng.standard_normal(heldout.shape)).astype(np.float32) for s in sigmas}


@torch.no_grad()
def evaluate_denoiser(model, heldout: np.ndarray, noisy: dict[float, np.ndarray], device) -> list[dict]:
    model.eval()
    rows = []
    for s, nz in noisy.items():
        x = torch.from_numpy(nz).unsqueeze(1).to(device)
        out = torch.cat([model(xc, s) for xc in torch.split(x, 24)]).squeeze(1).cpu().numpy()
        p_in = [psnr(nz[i], heldout[i]) for i in range(len(heldout))]
        p_out = [psnr(out[i], heldout[i]) for i in range(len(heldout))]
        rows.append({"sigma": s, "psnr_noisy": float(np.mean(p_in)), "psnr_denoised": float(np.mean(p_out)),
                     "gain_db": float(np.mean(p_out) - np.mean(p_in))})
    return rows


def train_denoiser(
    train_images: np.ndarray,
    heldout_images: np.ndarray,
    arch: dict,
    cfg: dict,
    out_dir: Path,
    device: torch.device,
    log=print,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SmallResidualDenoiser(depth=int(arch["depth"]), width=int(arch["width"])).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    n_iter = int(cfg["iterations"])
    lr0, lr_min = float(cfg["lr"]), float(cfg["lr_min"])
    smin, smax = (float(v) for v in cfg["sigma_range"])
    crop, bs = int(cfg["crop"]), int(cfg["batch_size"])
    eval_sigmas = [float(s) for s in cfg["eval_sigmas"]]
    noisy_heldout = _heldout_noise(heldout_images, eval_sigmas)
    train_t = torch.from_numpy(train_images)
    n, h, w = train_images.shape

    history = []
    init_eval = evaluate_denoiser(model, heldout_images, noisy_heldout, device)
    for r in init_eval:
        history.append({"iteration": 0, "train_loss": np.nan, **r})
    log(f"[denoiser] random-init held-out: {init_eval}")

    t0 = time.perf_counter()
    running = []
    for it in range(1, n_iter + 1):
        model.train()
        idx = rng.integers(0, n, size=bs)
        oy = rng.integers(0, h - crop + 1, size=bs)
        ox = rng.integers(0, w - crop + 1, size=bs)
        batch = torch.stack([train_t[i, a : a + crop, b : b + crop] for i, a, b in zip(idx, oy, ox)])
        k = int(rng.integers(0, 4))
        batch = torch.rot90(batch, k, dims=(-2, -1))
        if rng.random() < 0.5:
            batch = batch.flip(-1)
        x = batch.unsqueeze(1).to(device)
        sig = torch.from_numpy(rng.uniform(smin, smax, size=bs).astype(np.float32)).to(device)
        noisy = x + sig.view(-1, 1, 1, 1) * torch.randn(x.shape, generator=None, device=device)
        lr = lr_min + 0.5 * (lr0 - lr_min) * (1 + math.cos(math.pi * (it - 1) / max(1, n_iter - 1)))
        for g in opt.param_groups:
            g["lr"] = lr
        loss = torch.mean((model(noisy, sig) - x) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        running.append(float(loss.detach()))
        if it % int(cfg["eval_every"]) == 0 or it == n_iter:
            ev = evaluate_denoiser(model, heldout_images, noisy_heldout, device)
            tl = float(np.mean(running))
            running = []
            for r in ev:
                history.append({"iteration": it, "train_loss": tl, **r})
            gains = ", ".join(f"s={r['sigma']:.2f}:{r['gain_db']:+.2f}dB" for r in ev)
            log(f"[denoiser] it {it}/{n_iter} loss {tl:.2e} lr {lr:.1e} held-out gains {gains} ({time.perf_counter() - t0:.0f}s)")

    elapsed = time.perf_counter() - t0
    ckpt_path = out_dir / "model.pt"
    torch.save({"config": model.config(), "state_dict": model.state_dict()}, ckpt_path)
    hist = pd.DataFrame(history)
    hist.to_csv(out_dir / "train_log.csv", index=False)
    final = hist[hist.iteration == hist.iteration.max()]
    init = hist[hist.iteration == 0]
    heldout = init[["sigma", "psnr_noisy"]].copy()
    heldout["psnr_random_init"] = init["psnr_denoised"].to_numpy()
    heldout["psnr_trained"] = final["psnr_denoised"].to_numpy()
    heldout["gain_trained_db"] = heldout["psnr_trained"] - heldout["psnr_noisy"]
    heldout.to_csv(out_dir / "heldout.csv", index=False)
    meta = {
        "arch": model.config(),
        "n_parameters": model.n_parameters(),
        "train_cfg": cfg,
        "n_train_images": int(n),
        "n_heldout_images": int(len(heldout_images)),
        "train_seconds": elapsed,
        "device": str(device),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "improves_on_input_at_all_sigmas": bool((heldout["gain_trained_db"] > 0).all()),
    }
    save_json(meta, out_dir / "meta.json")
    return meta
