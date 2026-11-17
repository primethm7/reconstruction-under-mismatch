"""Experiment stages with caching and provenance.

Stage order and what each stage may read:

    data         -> generates train/val/test phantoms (ground truth)
    denoiser     -> train split (+ held-out val phantoms for monitoring only)
    observations -> ground truth + TRUE operator -> observations.npz (y only) + simulation.json
    tune         -> VALIDATION observations + validation ground truth (selection)
    run          -> TEST observations.npz ONLY (y), assumed operators, frozen selection
                    -> reconstructions + label-free scores. Never opens ground truth.
    evaluate     -> ground truth + reconstructions -> reference metrics, failure labels
    summarize / plot / report -> CSV outputs only

Cache keys contain the data hash, configuration subset, checkpoint hash and a hash of the
compute-relevant source code, so a stale result cannot be silently reused.
"""

from __future__ import annotations

import logging
import shutil
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..config import dump_yaml
from ..data.observations import (
    load_observations,
    load_simulation_meta,
    save_observations,
    simulate_observations,
)
from ..data.splits import build_synthetic_dataset, load_split_images
from ..denoisers import CNNDenoiser, load_cnn_denoiser
from ..denoisers.train import denoiser_key, train_denoiser
from ..metrics import image as im
from ..operators import GaussianBlur
from ..reconstruction import PnPADMM, tikhonov_fft
from .reconstruct import build_reconstructor, reconstruct_and_score  # noqa: F401
from ..utils.device import select_device
from ..utils.io import load_json, load_npz, save_json, save_npz, sha256_array, sha256_file, sha256_obj
from ..utils.provenance import PACKAGE_DIR, code_hash, environment, project_root

# --------------------------------------------------------------------------------------
# context
# --------------------------------------------------------------------------------------


def _file_code_hash(rel_paths: list[str]) -> str:
    import hashlib

    h = hashlib.sha256()
    for rel in rel_paths:
        h.update(rel.encode())
        h.update((PACKAGE_DIR / rel).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


DENOISER_CODE = ["denoisers/cnn.py", "denoisers/train.py"]
RUN_CODE = [
    "operators/blur.py",
    "reconstruction/tikhonov.py",
    "reconstruction/pnp_admm.py",
    "denoisers/cnn.py",
    "denoisers/ffdnet.py",
    "metrics/scores.py",
    "experiments/reconstruct.py",
]


@dataclass
class Context:
    cfg: dict
    root: Path = field(default_factory=project_root)
    device: torch.device | None = None
    device_log: dict | None = None
    log: logging.Logger | None = None

    @property
    def tag(self) -> str:
        """results/<profile>[_<results_tag>][<results_tag_suffix>] -- extensions never overwrite the main run."""
        tag = self.cfg["profile"]
        if self.cfg.get("results_tag"):
            tag += "_" + self.cfg["results_tag"]
        return tag + self.cfg.get("results_tag_suffix", "")

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def results(self) -> Path:
        return self.root / "results" / self.tag

    @property
    def runs(self) -> Path:
        return self.artifacts / "runs" / self.tag

    def setup(self) -> Context:
        self.results.mkdir(parents=True, exist_ok=True)
        (self.results / "logs").mkdir(exist_ok=True)
        torch.set_num_threads(int(self.cfg.get("num_threads", 4)))
        if self.device is None:
            self.device, self.device_log = select_device(self.cfg.get("device", "auto"))
        if self.log is None:
            lg = logging.getLogger(f"rum.{self.tag}")
            lg.setLevel(logging.INFO)
            lg.handlers.clear()
            fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
            sh = logging.StreamHandler(sys.stdout)
            sh.setFormatter(fmt)
            fh = logging.FileHandler(self.results / "logs" / "pipeline.log", encoding="utf-8")
            fh.setFormatter(fmt)
            lg.addHandler(sh)
            lg.addHandler(fh)
            lg.propagate = False
            self.log = lg
        dump_yaml(self.cfg, self.results / "config_resolved.yaml")
        save_json(environment(self.device_log), self.results / "environment.json")
        return self

    def info(self, msg: str) -> None:
        self.log.info(self.redact(msg))

    def redact(self, text: str) -> str:
        """Keep logs free of machine-specific absolute paths (user names)."""
        return str(text).replace(str(self.root), "<repo>").replace(str(Path.home()), "<home>")


def _status(ctx: Context, stage: str, status: str, **extra) -> None:
    path = ctx.results / "run_status.json"
    st = load_json(path) if path.exists() else {}
    st[stage] = {"status": status, "time": time.strftime("%Y-%m-%d %H:%M:%S"), **extra}
    save_json(st, path)


def run_stage(ctx: Context, name: str, fn, *args, **kwargs):
    _status(ctx, name, "running")
    t0 = time.perf_counter()
    try:
        out = fn(ctx, *args, **kwargs)
    except Exception as exc:
        ctx.info(f"[{name}] FAILED: {exc!r}")
        with open(ctx.results / "logs" / f"error_{name}.log", "w", encoding="utf-8") as f:
            f.write(ctx.redact(traceback.format_exc()))
        _status(ctx, name, "failed", error=repr(exc))
        raise
    _status(ctx, name, "done", seconds=round(time.perf_counter() - t0, 1))
    return out


# --------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------


def stage_data(ctx: Context) -> tuple[Path, dict]:
    src = ctx.cfg["data"].get("source", "synthetic")
    if src == "synthetic":
        ddir, manifest = build_synthetic_dataset(
            ctx.cfg["data"], ctx.cfg["master_seed"], ctx.cfg["image_size"], ctx.artifacts / "data"
        )
    elif src == "medmnist":
        from ..data.medmnist import build_medmnist_dataset

        ddir, manifest = build_medmnist_dataset(ctx.cfg["data"], ctx.cfg["master_seed"], ctx.artifacts, ctx.info)
    else:
        raise ValueError(f"unknown data source {src}")
    summary = {
        "key": manifest["key"],
        "spec": manifest["spec"],
        "splits": {s: {"n": v["n"], "sha256": v["sha256"]} for s, v in manifest["splits"].items()},
        "directory": ddir.relative_to(ctx.root).as_posix(),
    }
    if "source_info" in manifest:
        summary["source_info"] = manifest["source_info"]
    save_json(summary, ctx.results / "data_manifest.json")
    ctx.info(f"[data] {manifest['key']}: " + ", ".join(f"{s}={v['n']}" for s, v in manifest["splits"].items()))
    return ddir, manifest


# --------------------------------------------------------------------------------------
# denoiser
# --------------------------------------------------------------------------------------


def stage_denoiser(ctx: Context, ddir: Path, manifest: dict) -> tuple[Path, dict]:
    dcfg = ctx.cfg["denoiser"]
    synth_cfg = ctx.cfg.get("denoiser_training_data")  # extension configs reuse the synthetic model
    if synth_cfg:
        tddir, tman = build_synthetic_dataset(synth_cfg, ctx.cfg["master_seed"], ctx.cfg["image_size"], ctx.artifacts / "data")
    else:
        tddir, tman = ddir, manifest
    n_hold = int((synth_cfg or ctx.cfg["data"]).get("n_denoiser_heldout", 12))
    train = load_split_images(tddir, "train")
    heldout = load_split_images(tddir, "val")[:n_hold]
    key = denoiser_key(dcfg["arch"], dcfg["train"], tman["splits"]["train"]["sha256"], sha256_array(heldout),
                       _file_code_hash(DENOISER_CODE))
    out = ctx.artifacts / "checkpoints" / key
    committed = ctx.root / "results" / "checkpoints"
    if not (out / "meta.json").exists() and (committed / f"{key}_model.pt").exists():
        # a checkpoint committed to the repository with exactly this key (data + config + code hash)
        out.mkdir(parents=True, exist_ok=True)
        for src, dst in [("model.pt", "model.pt"), ("meta.json", "meta.json"), ("train_log.csv", "train_log.csv"),
                         ("heldout.csv", "heldout.csv")]:
            shutil.copy2(committed / f"{key}_{src}", out / dst)
        ctx.info(f"[denoiser] restored committed checkpoint results/checkpoints/{key}_model.pt (hash verified below)")
    if (out / "meta.json").exists() and (out / "model.pt").exists():
        meta = load_json(out / "meta.json")
        ctx.info(f"[denoiser] reusing cached checkpoint {key}")
    else:
        ctx.info(f"[denoiser] training {key} on {len(train)} synthetic train phantoms ({ctx.device})")
        meta = train_denoiser(train, heldout, dcfg["arch"], dcfg["train"], out, ctx.device, log=ctx.info)
    if meta["checkpoint_sha256"] != sha256_file(out / "model.pt"):
        raise RuntimeError("checkpoint hash mismatch")
    meta = {**meta, "key": key, "checkpoint": (out / "model.pt").relative_to(ctx.root).as_posix(),
            "trained_on": tman["key"]}
    dst = ctx.results / "denoiser"
    dst.mkdir(parents=True, exist_ok=True)
    for f in ["train_log.csv", "heldout.csv"]:
        shutil.copy2(out / f, dst / f)
    save_json(meta, dst / "meta.json")
    if not meta["improves_on_input_at_all_sigmas"]:
        ctx.info("[denoiser] WARNING: trained denoiser does not improve on the noisy input at every sigma")
    return out / "model.pt", meta


def build_denoiser(ctx: Context, ckpt: Path | None):
    kind = ctx.cfg["pnp"].get("denoiser", "small_cnn")
    chunk = int(ctx.cfg["pnp"].get("denoiser_chunk", 24))
    if kind == "small_cnn":
        den, _ = load_cnn_denoiser(ckpt, ctx.device)
        den.chunk = chunk
        return den, {"denoiser": "small_cnn", "checkpoint_sha256": sha256_file(ckpt)}
    if kind == "ffdnet_kair":
        from ..denoisers.ffdnet import load_kair_ffdnet

        den, info = load_kair_ffdnet(ctx.artifacts / "external" / "kair", ctx.device, log=ctx.info)
        den.chunk = chunk
        return den, info
    raise ValueError(f"unknown denoiser {kind}")


def stage_eval_pretrained(ctx: Context, denoiser) -> pd.DataFrame:
    """Held-out denoising check of a pretrained denoiser on the same synthetic phantoms and
    the same noise realisation used to monitor the small CNN (denoisers/train.py)."""
    from ..denoisers.train import _heldout_noise, evaluate_denoiser

    scfg = ctx.cfg.get("denoiser_training_data") or ctx.cfg["data"]
    if scfg.get("source", "synthetic") != "synthetic":
        scfg = {"n_train": 400, "n_val": 24, "n_test": 48, "n_denoiser_heldout": 12}
    tddir, _ = build_synthetic_dataset(scfg, ctx.cfg["master_seed"], ctx.cfg["image_size"], ctx.artifacts / "data")
    held = load_split_images(tddir, "val")[: int(scfg.get("n_denoiser_heldout", 12))]
    sigmas = [float(s) for s in ctx.cfg["denoiser"]["train"]["eval_sigmas"]]

    class _Wrap(torch.nn.Module):
        def __init__(self, d):
            super().__init__()
            self.d = d

        def forward(self, x, s):
            return self.d(x, s)

    rows = evaluate_denoiser(_Wrap(denoiser), held, _heldout_noise(held, sigmas), ctx.device)
    df = pd.DataFrame(rows)
    out = ctx.results / "denoiser"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"heldout_{denoiser.name}.csv", index=False)
    ctx.info(f"[denoiser] {denoiser.name} held-out gains: " + ", ".join(f"s={r['sigma']:.2f}:{r['gain_db']:+.2f}dB" for r in rows))
    return df


# --------------------------------------------------------------------------------------
# observations
# --------------------------------------------------------------------------------------


def stage_observations(ctx: Context, ddir: Path, manifest: dict, split: str) -> Path:
    op = ctx.cfg["operator"]
    seeds = ctx.cfg["noise"]["test_seeds" if split == "test" else "val_seeds"]
    spec = {"data": manifest["splits"][split]["sha256"], "sigma_true": op["sigma_true"], "truncate": op["truncate"],
            "noise": ctx.cfg["noise"]["std"], "seeds": seeds, "master_seed": ctx.cfg["master_seed"],
            "code": _file_code_hash(["data/observations.py", "operators/blur.py"])}
    out = ctx.runs / f"obs_{split}_{sha256_obj(spec)[:12]}"
    if not (out / "observations.npz").exists():
        images = load_split_images(ddir, split)
        obs, sim = simulate_observations(images, split, op["sigma_true"], ctx.cfg["noise"]["std"], seeds,
                                         ctx.cfg["master_seed"], op["truncate"])
        sim["spec"] = spec
        save_observations(obs, sim, out)
        ctx.info(f"[observations] {split}: {len(obs)} observations ({len(images)} images x {len(seeds)} noise seeds)")
    sim = load_simulation_meta(out)
    save_json({**sim, "directory": out.relative_to(ctx.root).as_posix()}, ctx.results / f"simulation_{split}.json")
    return out


# --------------------------------------------------------------------------------------
# tuning (validation only)
# --------------------------------------------------------------------------------------


def _psnr_batch(x: torch.Tensor, gt: np.ndarray) -> np.ndarray:
    xn = x.squeeze(1).cpu().numpy()
    return np.array([im.psnr(xn[i], gt[i]) for i in range(len(gt))])


def _argmax_first(df: pd.DataFrame, col: str) -> pd.Series:
    return df.loc[df[col].idxmax()]


def stage_tune(ctx: Context, ddir: Path, val_obs_dir: Path, denoiser, den_info: dict) -> dict:
    cfg = ctx.cfg
    obs = load_observations(val_obs_dir)
    sim = load_simulation_meta(val_obs_dir)  # validation: the matched (true) sigma is used by design
    gt = load_split_images(ddir, "val")[obs.image_index]
    spec = {"y": sha256_array(obs.y), "gt": sha256_array(gt), "tik": cfg["tikhonov"], "pnp": cfg["pnp"],
            "den": den_info, "failure": cfg["failure"], "code": _file_code_hash(RUN_CODE)}
    key = sha256_obj(spec)[:12]
    out = ctx.results / "tuning"
    sel_path = out / "selection.json"
    if sel_path.exists() and load_json(sel_path).get("key") == key:
        ctx.info(f"[tune] reusing selection {key}")
        return load_json(sel_path)
    out.mkdir(parents=True, exist_ok=True)
    dev = ctx.device
    y = obs.tensor(dev)
    op = GaussianBlur(sim["sigma_true"], y.shape[-2:], truncate=cfg["operator"]["truncate"], device=dev)

    # Tikhonov: lambda grid
    rows = []
    for lam in cfg["tikhonov"]["lambda_grid"]:
        x = tikhonov_fft(y, op, lam).clamp(0, 1)
        p = _psnr_batch(x, gt)
        rows.append({"lambda": lam, "psnr_mean": p.mean(), "psnr_median": np.median(p)})
    tik = pd.DataFrame(rows)
    tik.to_csv(out / "tuning_tikhonov.csv", index=False)
    best_t = _argmax_first(tik, "psnr_mean")
    lam = float(best_t["lambda"])
    ctx.info(f"[tune] Tikhonov lambda={lam:.3g} val PSNR={best_t['psnr_mean']:.2f}")

    # PnP-ADMM: rho x sigma_d x K (K read off snapshots of one run of max(K) iterations)
    pcfg = cfg["pnp"]
    iters = sorted(int(k) for k in pcfg["iters_grid"])
    rows = []
    t0 = time.perf_counter()
    for rho in pcfg["rho_grid"]:
        for sd in pcfg["sigma_d_grid"]:
            pnp = PnPADMM(denoiser, rho, sd, max(iters), clip=pcfg["clip_output"],
                          divergence_bound=pcfg["divergence_bound"], denoiser_name=den_info["denoiser"])
            snaps: dict[int, list[torch.Tensor]] = {k: [] for k in iters}
            n_div = 0
            for yb in torch.split(y, int(pcfg["batch_size"])):
                _, info = pnp.run(yb, op, record_at=iters)
                n_div += int(info["diverged"].sum())
                for k in iters:
                    snaps[k].append(info["snapshots"][k])
            for k in iters:
                p = _psnr_batch(torch.cat(snaps[k]), gt)
                rows.append({"rho": rho, "sigma_d": sd, "n_iter": k, "psnr_mean": p.mean(),
                             "psnr_median": np.median(p), "n_diverged": n_div})
            ctx.info(f"[tune] PnP rho={rho} sigma_d={sd}: PSNR@K " +
                     " ".join(f"{k}:{r['psnr_mean']:.2f}" for k, r in zip(iters, rows[-len(iters):])) +
                     f" ({time.perf_counter() - t0:.0f}s)")
    pnp_df = pd.DataFrame(rows)
    pnp_df.to_csv(out / "tuning_pnp.csv", index=False)
    # tie-break: highest mean PSNR; among exact ties the smallest K (rows are ordered by K)
    best_p = pnp_df.sort_values(["psnr_mean", "n_iter"], ascending=[False, True]).iloc[0]

    def _boundary(val, grid):
        return bool(val == min(grid) or val == max(grid))

    sel = {
        "key": key,
        "selection_metric": "mean PSNR on validation observations, matched operator (sigma_assumed = sigma_true)",
        "validation": {"n_images": int(len(np.unique(obs.image_index))), "noise_seeds": sim["noise_seeds"],
                       "n_observations": int(len(obs))},
        "tikhonov": {"lam": lam, "val_psnr_mean": float(best_t["psnr_mean"]),
                     "on_grid_boundary": _boundary(lam, cfg["tikhonov"]["lambda_grid"])},
        "pnp_admm": {"rho": float(best_p["rho"]), "sigma_d": float(best_p["sigma_d"]), "n_iter": int(best_p["n_iter"]),
                     "val_psnr_mean": float(best_p["psnr_mean"]), "denoiser": den_info,
                     "on_grid_boundary": {"rho": _boundary(best_p["rho"], pcfg["rho_grid"]),
                                          "sigma_d": _boundary(best_p["sigma_d"], pcfg["sigma_d_grid"]),
                                          "n_iter": _boundary(best_p["n_iter"], iters)}},
    }
    ctx.info(f"[tune] PnP selected rho={sel['pnp_admm']['rho']} sigma_d={sel['pnp_admm']['sigma_d']} "
             f"K={sel['pnp_admm']['n_iter']} val PSNR={best_p['psnr_mean']:.2f}")

    # failure thresholds from validation matched PSNR with the selected settings
    margin = float(cfg["failure"]["margin_db"])
    thr = {}
    val_rows = []
    for method in cfg["methods"]:
        rec = build_reconstructor(method, sel, denoiser, cfg)
        xs = [rec.reconstruct(yb, op)[0] for yb in torch.split(y, int(pcfg["batch_size"]))]
        p = _psnr_batch(torch.cat(xs), gt)
        med = float(np.median(p))
        thr[method] = {"median_val_matched_psnr": med, "threshold_psnr": med - margin}
        val_rows += [{"method": method, "image_index": int(i), "noise_seed": int(s), "psnr": float(v)}
                     for i, s, v in zip(obs.image_index, obs.noise_seed, p)]
    pd.DataFrame(val_rows).to_csv(out / "validation_matched_psnr.csv", index=False)
    sel["failure"] = {"rule": f"PSNR < median(validation matched PSNR of the same method) - {margin} dB",
                      "margin_db": margin, "thresholds": thr,
                      "relative_worst_fraction": float(cfg["failure"]["relative_worst_fraction"])}
    sel["frozen_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_json(sel, sel_path)
    for m, t in thr.items():
        ctx.info(f"[tune] failure threshold {m}: PSNR < {t['threshold_psnr']:.2f} dB")
    return sel


# --------------------------------------------------------------------------------------
# run (test observations only -- no ground truth)
# --------------------------------------------------------------------------------------


def stage_run(ctx: Context, test_obs_dir: Path, sel: dict, denoiser) -> Path:
    cfg = ctx.cfg
    obs = load_observations(test_obs_dir)  # y only; simulation.json is NOT read here
    y = obs.tensor(ctx.device)
    sel_core = {k: v for k, v in sel.items() if k in ("tikhonov", "pnp_admm")}
    spec = {"y": sha256_array(obs.y), "sel": sel_core, "sigma_assumed": cfg["operator"]["sigma_assumed"],
            "truncate": cfg["operator"]["truncate"], "scores": cfg["scores"], "methods": cfg["methods"],
            "clip": [cfg["tikhonov"]["clip_output"], cfg["pnp"]["clip_output"]], "code": _file_code_hash(RUN_CODE)}
    key = sha256_obj(spec)[:12]
    run_dir = ctx.runs / f"run_{key}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(spec, run_dir / "spec.json")
    for method in cfg["methods"]:
        rec = build_reconstructor(method, sel, denoiser, cfg)
        for sa in cfg["operator"]["sigma_assumed"]:
            chunk = run_dir / f"{method}__sa{sa:.4f}.npz"
            if chunk.exists():
                ctx.info(f"[run] reuse {chunk.name}")
                continue
            t0 = time.perf_counter()
            res = reconstruct_and_score(y, rec, sa, cfg["operator"]["truncate"], cfg["scores"], int(cfg["pnp"]["batch_size"]))
            save_npz(chunk, **res)
            ctx.info(f"[run] {method} sigma_assumed={sa:.4f}: {len(obs)} obs in {time.perf_counter() - t0:.1f}s, "
                     f"diverged={int(res['diverged'].sum())}")
    save_json({"run_dir": run_dir.relative_to(ctx.root).as_posix(), "key": key, "spec": spec,
               "reconstructors": {m: build_reconstructor(m, sel, denoiser, cfg).describe() for m in cfg["methods"]}},
              ctx.results / "run_manifest.json")
    return run_dir


# --------------------------------------------------------------------------------------
# evaluate (ground truth enters here)
# --------------------------------------------------------------------------------------


def stage_evaluate(ctx: Context, ddir: Path, test_obs_dir: Path, run_dir: Path, sel: dict) -> pd.DataFrame:
    cfg = ctx.cfg
    obs = load_observations(test_obs_dir)
    sim = load_simulation_meta(test_obs_dir)
    gt_all = load_split_images(ddir, "test")
    rows = []
    for method in cfg["methods"]:
        thr = sel["failure"]["thresholds"][method]["threshold_psnr"]
        for sa in cfg["operator"]["sigma_assumed"]:
            d = load_npz(run_dir / f"{method}__sa{sa:.4f}.npz")
            ratio = round(sa / sim["sigma_true"], 6)
            for i in range(len(obs)):
                gt = gt_all[obs.image_index[i]]
                x = d["x_hat"][i]
                p = im.psnr(x, gt)
                rows.append({
                    "image_id": str(obs.image_ids[i]),
                    "image_index": int(obs.image_index[i]),
                    "noise_seed": int(obs.noise_seed[i]),
                    "method": method,
                    "ratio": ratio,
                    "sigma_assumed": sa,
                    "psnr": p,
                    "ssim": im.ssim(x, gt),
                    "mse": im.mse(x, gt),
                    "rel_error": im.relative_error(x, gt),
                    "psnr_unclipped": im.psnr(d["x_raw"][i], gt),
                    "rel_residual": float(d["rel_residual"][i]),
                    "residual_acf": float(d["residual_acf"][i]),
                    "sensitivity": float(d["sensitivity"][i]),
                    "runtime_s": float(d["runtime_s"][i]),
                    "sens_runtime_s": float(d["sens_runtime_s"][i]),
                    "iterations": int(d["iterations"][i]),
                    "final_rel_change": float(d["final_rel_change"][i]),
                    "diverged": bool(d["diverged"][i]),
                    "nonfinite_output": bool(not np.isfinite(x).all()),
                    "failure": bool(p < thr),
                })
    df = pd.DataFrame(rows)
    frac = float(sel["failure"]["relative_worst_fraction"])
    df["relative_failure"] = False
    for _, g in df.groupby(["method", "ratio", "noise_seed"]):
        k = int(np.ceil(frac * len(g) - 1e-9))
        worst = g.sort_values(["mse", "image_index"], ascending=[False, True]).index[:k]
        df.loc[worst, "relative_failure"] = True
    df.to_csv(ctx.results / "per_image.csv", index=False)
    ctx.info(f"[evaluate] wrote per_image.csv ({len(df)} rows); absolute failures per method: "
             + ", ".join(f"{m}={int(g.failure.sum())}" for m, g in df.groupby("method")))
    return df
