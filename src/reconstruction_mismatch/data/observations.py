"""Simulated observations y = A_{sigma_true} x + eps and the ground-truth boundary.

`simulate_observations` is the only place where ground truth and the true operator meet.
It writes two files:

* observations.npz -- y, image index, noise seed. This is ALL the reconstruction stage sees.
* simulation.json  -- sigma_true, noise std, seeds. Read only by evaluation / reporting.

The noise realisation depends only on (master_seed, split, image index, noise seed), never
on the method or the assumed sigma, so every method and every sigma_assumed is applied to
exactly the same y.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..operators import GaussianBlur
from ..utils.io import load_json, load_npz, save_json, save_npz, sha256_array
from .splits import SPLIT_CODES

NOISE_STREAM_OFFSET = 100


@dataclass
class ObservationSet:
    """Observation-side data only: no ground truth, no true operator."""

    y: np.ndarray  # (N, H, W) float32
    image_index: np.ndarray  # (N,)
    noise_seed: np.ndarray  # (N,)
    image_ids: np.ndarray  # (N,) strings like "test-0007"

    def tensor(self, device="cpu", dtype=torch.float32) -> torch.Tensor:
        return torch.from_numpy(self.y).to(device=device, dtype=dtype).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.y)


def noise_rng(master_seed: int, split: str, index: int, noise_seed: int) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence([int(master_seed), NOISE_STREAM_OFFSET + SPLIT_CODES[split], int(index), int(noise_seed)])
    )


def simulate_observations(
    images: np.ndarray,
    split: str,
    sigma_true: float,
    noise_std: float,
    noise_seeds: list[int],
    master_seed: int,
    truncate: float,
    id_prefix: str | None = None,
) -> tuple[ObservationSet, dict]:
    n, h, w = images.shape
    op = GaussianBlur(sigma_true, (h, w), truncate=truncate, dtype=torch.float64)
    clean_blur = op(torch.from_numpy(images.astype(np.float64)).unsqueeze(1)).squeeze(1).numpy()
    ys, idx, seeds = [], [], []
    for s in noise_seeds:
        for i in range(n):
            eps = noise_rng(master_seed, split, i, s).standard_normal((h, w))
            ys.append((clean_blur[i] + noise_std * eps).astype(np.float32))
            idx.append(i)
            seeds.append(s)
    prefix = id_prefix or split
    obs = ObservationSet(
        y=np.stack(ys),
        image_index=np.asarray(idx),
        noise_seed=np.asarray(seeds),
        image_ids=np.asarray([f"{prefix}-{i:04d}" for i in idx]),
    )
    sim = {
        "split": split,
        "sigma_true": float(sigma_true),
        "noise_std": float(noise_std),
        "noise_seeds": [int(s) for s in noise_seeds],
        "truncate": float(truncate),
        "boundary": "circular",
        "y_clipped": False,
        "n_images": int(n),
        "y_sha256": sha256_array(obs.y),
    }
    return obs, sim


def save_observations(obs: ObservationSet, sim: dict, out_dir: Path) -> None:
    save_npz(Path(out_dir) / "observations.npz", y=obs.y, image_index=obs.image_index,
             noise_seed=obs.noise_seed, image_ids=obs.image_ids.astype("U32"))
    save_json(sim, Path(out_dir) / "simulation.json")


def load_observations(out_dir: Path) -> ObservationSet:
    """Reconstruction-side loader: reads observations.npz only (never simulation.json)."""
    d = load_npz(Path(out_dir) / "observations.npz")
    return ObservationSet(y=d["y"], image_index=d["image_index"], noise_seed=d["noise_seed"], image_ids=d["image_ids"])


def load_simulation_meta(out_dir: Path) -> dict:
    """Evaluation-side loader for the true simulation parameters."""
    return load_json(Path(out_dir) / "simulation.json")
