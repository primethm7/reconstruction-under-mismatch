"""OrganCMNIST (MedMNIST+, 128x128) -- optional public biomedical image extension.

Source   : Zenodo record 10519652, "MedMNIST+ 18x Standardized Datasets ..." v3.0,
           DOI 10.5281/zenodo.10519652, file organcmnist_128.npz (287.5 MB,
           md5 773c1f009daa3fe5d9a2a201b2a7ed94). License: CC BY 4.0.
Origin   : abdominal CT from the LiTS benchmark (Bilic et al. 2023) with organ boxes from
           Xu et al. 2019; coronal-view 2-D crops, abdominal HU window, grey-scale uint8.
Split    : the official MedMNIST split is by CT scan -- 115 / 16 scans (from the LiTS
           training set) for train / val, 70 scans (LiTS test set) for test (Yang et al.,
           Sci. Data 2023). Per-slice patient IDs are NOT released, so we keep the official
           split and cannot cluster slices of the same scan within a split.
Use here : a deterministic random subset of the official VAL split (tuning, thresholds) and
           TEST split (evaluation); intensities /255 -> [0, 1]. Blur and noise are then
           simulated: "public biomedical image based synthetic observation experiment" --
           not a real CT measurement and not a clinical validation. Nothing is trained on
           these images (the PnP denoiser is the synthetic-data model).
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

from ..utils.io import md5_file, save_json, save_npz, sha256_array, sha256_obj

DATASETS = {
    "organcmnist_128": {
        "url": "https://zenodo.org/records/10519652/files/organcmnist_128.npz?download=1",
        "md5": "773c1f009daa3fe5d9a2a201b2a7ed94",
        "size_mb": 287.5,
        "record": "https://zenodo.org/records/10519652",
        "doi": "10.5281/zenodo.10519652",
        "version": "MedMNIST+ v3.0 (Zenodo, 2024-01-16)",
        "license": "CC BY 4.0",
        "citation": "Yang et al., MedMNIST v2, Scientific Data 10, 41 (2023); Bilic et al., LiTS, Medical Image "
                    "Analysis 84 (2023); Xu et al., IEEE TMI 38(8) (2019)",
        "split_unit": "CT scan (official: 115/16/70 scans for train/val/test); per-slice scan IDs not released",
        "modality": "abdominal CT, coronal 2-D crops, abdominal window, uint8",
    }
}
SPLIT_SUBSET_CODE = {"val": 21, "test": 31}


def download(name: str, dest_dir: Path, log=print) -> Path:
    info = DATASETS[name]
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"{name}.npz"
    if path.exists() and md5_file(path) == info["md5"]:
        return path
    log(f"[medmnist] downloading {info['url']} ({info['size_mb']} MB) -> {path}")
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(info["url"], tmp)
    got = md5_file(tmp)
    if got != info["md5"]:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"md5 mismatch for {name}: {got} != {info['md5']}")
    tmp.replace(path)
    return path


def build_medmnist_dataset(data_cfg: dict, master_seed: int, artifacts: Path, log=print) -> tuple[Path, dict]:
    name = data_cfg["dataset"]
    info = DATASETS[name]
    spec = {"source": "medmnist", "dataset": name, "md5": info["md5"], "seed": int(master_seed),
            "n_val": int(data_cfg["n_val"]), "n_test": int(data_cfg["n_test"])}
    key = "medmnist-" + sha256_obj(spec)[:12]
    ddir = Path(artifacts) / "data" / key
    if (ddir / "manifest.json").exists():
        from ..utils.io import load_json

        return ddir, load_json(ddir / "manifest.json")
    raw = download(name, Path(artifacts) / "external" / "medmnist", log=log)
    manifest = {"key": key, "spec": spec, "splits": {}, "source_info": {**info, "file_md5_verified": True}}
    with np.load(raw, allow_pickle=False) as d:
        for split in ["val", "test"]:
            all_imgs = d[f"{split}_images"]
            n = int(data_cfg[f"n_{split}"])
            rng = np.random.default_rng(np.random.SeedSequence([int(master_seed), SPLIT_SUBSET_CODE[split]]))
            sel = np.sort(rng.choice(len(all_imgs), size=n, replace=False))
            imgs = (all_imgs[sel].astype(np.float32) / 255.0)
            if imgs.ndim == 4:  # (N, H, W, 1) safety
                imgs = imgs[..., 0]
            save_npz(ddir / f"{split}.npz", images=imgs, index=np.arange(n), source_index=sel)
            manifest["splits"][split] = {"n": n, "sha256": sha256_array(imgs),
                                         "image_sha256": [sha256_array(x) for x in imgs],
                                         "source_split_size": int(len(all_imgs)), "source_indices": sel.tolist()}
    save_npz(ddir / "train.npz", images=np.zeros((0, 128, 128), np.float32), index=np.zeros(0, int))
    manifest["splits"]["train"] = {"n": 0, "sha256": sha256_array(np.zeros((0, 128, 128), np.float32)), "image_sha256": []}
    save_json(manifest, ddir / "manifest.json")
    log(f"[medmnist] {name}: val {data_cfg['n_val']} / test {data_cfg['n_test']} slices sampled from the official split")
    return ddir, manifest
