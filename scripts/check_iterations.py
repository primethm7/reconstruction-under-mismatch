"""Validation-only check: does PnP-ADMM keep improving beyond the largest K in the grid?

Runs the selected (rho, sigma_d) for 100 iterations on the VALIDATION observations of a
profile (matched operator) and saves PSNR at K in {30, 40, 60, 80, 100} to
results/<profile>/tuning/iteration_check.csv. Test data are not touched.

    python scripts/check_iterations.py --profile full
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from reconstruction_mismatch.data.observations import load_observations, load_simulation_meta
from reconstruction_mismatch.data.splits import load_split_images
from reconstruction_mismatch.denoisers import load_cnn_denoiser
from reconstruction_mismatch.metrics.image import psnr
from reconstruction_mismatch.operators import GaussianBlur
from reconstruction_mismatch.reconstruction import PnPADMM
from reconstruction_mismatch.utils.io import load_json
from reconstruction_mismatch.utils.provenance import project_root


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="full")
    args = ap.parse_args()
    root = project_root()
    res = root / "results" / args.profile
    torch.set_num_threads(4)
    obs_dir = root / load_json(res / "simulation_val.json")["directory"]
    obs = load_observations(obs_dir)
    sigma_true = load_simulation_meta(obs_dir)["sigma_true"]
    gt = load_split_images(root / load_json(res / "data_manifest.json")["directory"], "val")[obs.image_index]
    sel = load_json(res / "tuning" / "selection.json")["pnp_admm"]
    den, _ = load_cnn_denoiser(root / load_json(res / "denoiser" / "meta.json")["checkpoint"])
    ks = [30, 40, 60, 80, 100]
    _, info = PnPADMM(den, sel["rho"], sel["sigma_d"], max(ks)).run(obs.tensor(), GaussianBlur(sigma_true, (128, 128)),
                                                                     record_at=ks)
    rows = []
    for k in ks:
        p = np.mean([psnr(info["snapshots"][k][i, 0].numpy(), gt[i]) for i in range(len(gt))])
        rows.append({"n_iter": k, "val_psnr_mean": p})
    df = pd.DataFrame(rows)
    df["psnr_minus_k40"] = df.val_psnr_mean - float(df.loc[df.n_iter == 40, "val_psnr_mean"].iloc[0])
    df["rho"], df["sigma_d"] = sel["rho"], sel["sigma_d"]
    df["final_rel_change_median_at_100"] = float(np.median(info["final_rel_change"]))
    out = res / "tuning" / "iteration_check.csv"
    df.to_csv(out, index=False)
    print(df.round(4).to_string(index=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
