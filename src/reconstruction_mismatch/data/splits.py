"""Independent train / val / test splits of synthetic phantoms.

Every image has its own random stream, seeded by (master_seed, split_code, index) through
numpy's SeedSequence, so the three splits are generated independently (no crops or noise
repeats of the same original are shared across splits), and the first n images of a split
are identical whatever n is (quick is a subset of full).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..utils.io import load_json, load_npz, save_json, save_npz, sha256_array, sha256_obj
from .synthetic import GENERATOR_VERSION, generate_phantom

SPLIT_CODES = {"train": 1, "val": 2, "test": 3}


@dataclass
class SplitInfo:
    name: str
    n: int
    sha256: str
    image_sha256: list[str]


def image_rng(master_seed: int, split: str, index: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(master_seed), SPLIT_CODES[split], int(index)]))


def generate_split(split: str, n: int, master_seed: int, size: int) -> np.ndarray:
    out = np.empty((n, size, size), dtype=np.float32)
    for i in range(n):
        out[i] = generate_phantom(image_rng(master_seed, split, i), size=size)
    return out


def dataset_key(data_cfg: dict, master_seed: int, size: int) -> str:
    spec = {
        "generator": GENERATOR_VERSION,
        "seed": int(master_seed),
        "size": int(size),
        "n_train": int(data_cfg["n_train"]),
        "n_val": int(data_cfg["n_val"]),
        "n_test": int(data_cfg["n_test"]),
    }
    return "synth-" + sha256_obj(spec)[:12], spec


def build_synthetic_dataset(data_cfg: dict, master_seed: int, size: int, root: Path) -> tuple[Path, dict]:
    """Generate (or reuse) the dataset; returns its directory and manifest."""
    key, spec = dataset_key(data_cfg, master_seed, size)
    ddir = Path(root) / key
    manifest_path = ddir / "manifest.json"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        if all((ddir / f"{s}.npz").exists() for s in SPLIT_CODES):
            return ddir, manifest
    manifest = {"key": key, "spec": spec, "splits": {}}
    for split in SPLIT_CODES:
        n = int(data_cfg[f"n_{split}"])
        imgs = generate_split(split, n, master_seed, size)
        save_npz(ddir / f"{split}.npz", images=imgs, index=np.arange(n))
        manifest["splits"][split] = {
            "n": n,
            "sha256": sha256_array(imgs),
            "image_sha256": [sha256_array(im) for im in imgs],
            "seed_rule": f"SeedSequence([{master_seed}, {SPLIT_CODES[split]}, index])",
        }
    check_no_overlap(manifest)
    save_json(manifest, manifest_path)
    return ddir, manifest


def check_no_overlap(manifest: dict) -> None:
    seen: dict[str, str] = {}
    for split, info in manifest["splits"].items():
        for i, h in enumerate(info["image_sha256"]):
            if h in seen:
                raise RuntimeError(f"image {split}[{i}] duplicates {seen[h]}")
            seen[h] = f"{split}[{i}]"


def load_split_images(ddir: Path, split: str) -> np.ndarray:
    """Ground-truth images. Only data generation, denoiser training and *evaluation* call this."""
    return load_npz(Path(ddir) / f"{split}.npz")["images"]
