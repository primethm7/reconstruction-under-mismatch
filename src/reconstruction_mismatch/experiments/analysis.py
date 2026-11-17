"""Summaries, failure-capture analysis and image-level cluster bootstrap.

Unit of resampling: the ORIGINAL test image. Noise seeds and sigma_assumed conditions of
the same image are kept together in every bootstrap resample (paired / cluster bootstrap),
so they are never counted as independent samples. With 48 images the intervals are
exploratory, not confirmatory.

A. within-condition: rank the n images of one (method, ratio, noise seed); average the
   capture over noise seeds.
B. pooled: rank all 5n reconstructions of one (method, noise seed) across ratios.
   Because sigma_true is FIXED in this experiment, sigma_assumed alone identifies the
   mismatch level; a pooled score can look good just by tracking the condition. The
   'condition-only' diagnostic replaces each score by its mean over the condition (a
   label-free quantity that carries no within-condition information). If the real score
   is no better than its condition-only version, its pooled performance reflects condition
   differences, not per-image discrimination.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..metrics.failure import auroc, capture_rate, oracle_capture, random_capture, review_k, spearman
from ..utils.io import save_json

SCORES = ["rel_residual", "residual_acf", "sensitivity"]
SCORE_LABELS = {"rel_residual": "relative residual", "residual_acf": "residual autocorrelation",
                "sensitivity": "operator sensitivity"}


# --------------------------------------------------------------------------------------
# helpers: cube[method] -> arrays of shape (n_img, n_seed, n_ratio)
# --------------------------------------------------------------------------------------


def _cubes(df: pd.DataFrame, cols: list[str]) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    imgs = np.sort(df.image_index.unique())
    seeds = np.sort(df.noise_seed.unique())
    ratios = np.sort(df.ratio.unique())
    out = {}
    for m, g in df.groupby("method"):
        g = g.set_index(["image_index", "noise_seed", "ratio"]).sort_index()
        idx = pd.MultiIndex.from_product([imgs, seeds, ratios])
        g = g.reindex(idx)
        out[m] = {c: g[c].to_numpy().reshape(len(imgs), len(seeds), len(ratios)) for c in cols}
    return out, imgs, seeds, ratios


def _ci(samples: np.ndarray, ci: float) -> tuple[float, float]:
    s = samples[np.isfinite(samples)]
    if len(s) == 0:
        return float("nan"), float("nan")
    a = (1 - ci) / 2
    return float(np.quantile(s, a)), float(np.quantile(s, 1 - a))


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


def summarize(results: Path, cfg: dict, log=print) -> dict:
    results = Path(results)
    df = pd.read_csv(results / "per_image.csv")
    fracs = [float(f) for f in cfg["review_fractions"]]
    bcfg = cfg["bootstrap"]
    rng = np.random.default_rng(int(bcfg["seed"]))
    B = int(bcfg["n_resamples"])
    ci = float(bcfg["ci"])
    methods = list(cfg["methods"])
    cols = ["psnr", "ssim", "mse", "rel_error", "failure", "relative_failure"] + SCORES
    cubes, imgs, seeds, ratios = _cubes(df, cols)
    n_img = len(imgs)
    boot_idx = rng.integers(0, n_img, size=(B, n_img))

    # ---------------- 1. performance by condition -------------------------------------
    rows = []
    for m in methods:
        c = cubes[m]
        for j, r in enumerate(ratios):
            g = df[(df.method == m) & (df.ratio == r)]
            row = {"method": m, "ratio": r, "sigma_assumed": float(g.sigma_assumed.iloc[0]),
                   "n_images": n_img, "n_seeds": len(seeds)}
            for metric in ["psnr", "ssim", "rel_error"]:
                per_img = np.nanmean(c[metric][:, :, j], axis=1)
                bs = per_img[boot_idx].mean(axis=1)
                lo, hi = _ci(bs, ci)
                row.update({f"{metric}_mean": per_img.mean(), f"{metric}_sd_images": per_img.std(ddof=1) if n_img > 1 else np.nan,
                            f"{metric}_ci_lo": lo, f"{metric}_ci_hi": hi})
            row.update({
                "psnr_unclipped_mean": g.psnr_unclipped.mean(),
                "runtime_s_per_image": g.runtime_s.mean(),
                "sens_runtime_s_per_image": g.sens_runtime_s.mean(),
                "iterations": g.iterations.mean(),
                "final_rel_change_median": g.final_rel_change.median(),
                "n_diverged": int(g.diverged.sum()),
                "n_nonfinite": int(g.nonfinite_output.sum()),
                "failure_rate": g.failure.mean(),
                "n_failures": int(g.failure.sum()),
                "n_obs": len(g),
                **{f"{s}_mean": g[s].mean() for s in SCORES},
            })
            rows.append(row)
    perf = pd.DataFrame(rows)
    perf.to_csv(results / "summary_by_condition.csv", index=False)

    # ---------------- 2. paired method difference -------------------------------------
    rows = []
    if "tikhonov" in cubes and "pnp_admm" in cubes:
        for j, r in enumerate(ratios):
            for metric in ["psnr", "ssim"]:
                d = np.nanmean(cubes["pnp_admm"][metric][:, :, j] - cubes["tikhonov"][metric][:, :, j], axis=1)
                lo, hi = _ci(d[boot_idx].mean(axis=1), ci)
                rows.append({"ratio": r, "metric": metric, "pnp_minus_tikhonov_mean": d.mean(), "ci_lo": lo, "ci_hi": hi,
                             "frac_images_pnp_better": float((d > 0).mean())})
    pd.DataFrame(rows).to_csv(results / "method_difference.csv", index=False)

    # ---------------- 2b. paired difference of each ratio vs the matched condition ------
    rows = []
    j1 = int(np.argmin(np.abs(ratios - 1.0)))
    for m in methods:
        for j, r in enumerate(ratios):
            d = np.nanmean(cubes[m]["psnr"][:, :, j] - cubes[m]["psnr"][:, :, j1], axis=1)
            lo, hi = _ci(d[boot_idx].mean(axis=1), ci)
            rows.append({"method": m, "ratio": r, "psnr_minus_matched_mean": d.mean(), "ci_lo": lo, "ci_hi": hi,
                         "frac_images_better_than_matched": float((d > 0).mean())})
    pd.DataFrame(rows).to_csv(results / "ratio_vs_matched.csv", index=False)

    # ---------------- 3. failure capture: within condition (A) ------------------------
    rows = []
    for m in methods:
        c = cubes[m]
        for label in ["failure", "relative_failure"]:
            for j, r in enumerate(ratios):
                for f in fracs:
                    per_seed = {s: [] for s in SCORES + ["random", "oracle"]}
                    n_fail = []
                    for si in range(len(seeds)):
                        lab = c[label][:, si, j].astype(bool)
                        n_fail.append(int(lab.sum()))
                        for s in SCORES:
                            per_seed[s].append(capture_rate(c[s][:, si, j], lab, f))
                        per_seed["random"].append(random_capture(n_img, f) if lab.sum() else np.nan)
                        per_seed["oracle"].append(oracle_capture(lab, f))
                    for s, vals in per_seed.items():
                        vals = np.array(vals, float)
                        rows.append({"method": m, "label": label, "ratio": r, "fraction": f,
                                     "k": review_k(n_img, f), "n": n_img, "score": s,
                                     "capture": np.nanmean(vals) if np.isfinite(vals).any() else np.nan,
                                     "n_seeds_defined": int(np.isfinite(vals).sum()),
                                     "failures_per_seed": "/".join(map(str, n_fail))})
    within = pd.DataFrame(rows)
    within.to_csv(results / "failure_capture_within.csv", index=False)

    # ---------------- 4. failure capture: pooled across ratios (B) --------------------
    rows = []
    for m in methods:
        c = cubes[m]
        for label in ["failure", "relative_failure"]:
            for f in fracs:
                acc: dict[str, list] = {}
                n_fail = []
                for si in range(len(seeds)):
                    lab = c[label][:, si, :].reshape(-1).astype(bool)
                    n_fail.append(int(lab.sum()))
                    n = lab.size
                    for s in SCORES:
                        sc = c[s][:, si, :]
                        acc.setdefault(s, []).append(capture_rate(sc.reshape(-1), lab, f))
                        cond = np.broadcast_to(np.nanmean(sc, axis=0, keepdims=True), sc.shape).reshape(-1)
                        acc.setdefault(f"{s}[condition-only]", []).append(capture_rate(cond, lab, f))
                    acc.setdefault("random", []).append(random_capture(n, f) if lab.sum() else np.nan)
                    acc.setdefault("oracle", []).append(oracle_capture(lab, f))
                for s, vals in acc.items():
                    vals = np.array(vals, float)
                    rows.append({"method": m, "label": label, "fraction": f, "k": review_k(n_img * len(ratios), f),
                                 "n": n_img * len(ratios), "score": s,
                                 "capture": np.nanmean(vals) if np.isfinite(vals).any() else np.nan,
                                 "n_seeds_defined": int(np.isfinite(vals).sum()),
                                 "failures_per_seed": "/".join(map(str, n_fail))})
    pooled = pd.DataFrame(rows)
    pooled.to_csv(results / "failure_capture_pooled.csv", index=False)

    # ---------------- 5. score association with bootstrap CIs -------------------------
    def _assoc(c, s, j=None, mode="spearman"):
        """Mean over seeds of Spearman(score, MSE) or AUROC(score -> failure)."""
        vals = []
        for si in range(len(seeds)):
            if j is None:
                sc, mse, lab = c[s][:, si, :].reshape(-1), c["mse"][:, si, :].reshape(-1), c["failure"][:, si, :].reshape(-1)
            else:
                sc, mse, lab = c[s][:, si, j], c["mse"][:, si, j], c["failure"][:, si, j]
            vals.append(spearman(sc, mse) if mode == "spearman" else auroc(sc, lab.astype(bool)))
        vals = np.array(vals, float)
        return np.nanmean(vals) if np.isfinite(vals).any() else np.nan

    rows = []
    for m in methods:
        c = cubes[m]
        for s in SCORES:
            for scope, j_list in [("within", list(range(len(ratios)))), ("pooled", [None])]:
                for j in j_list:
                    for mode in ["spearman", "auroc"]:
                        est = _assoc(c, s, j, mode)
                        bs = []
                        for b in range(B if mode == "spearman" else min(B, 500)):
                            cb = {k: v[boot_idx[b]] for k, v in c.items()}
                            bs.append(_assoc(cb, s, j, mode))
                        lo, hi = _ci(np.array(bs, float), ci)
                        rows.append({"method": m, "score": s, "scope": scope,
                                     "ratio": ratios[j] if j is not None else "all",
                                     "statistic": "spearman_vs_mse" if mode == "spearman" else "auroc_failure",
                                     "estimate": est, "ci_lo": lo, "ci_hi": hi})
    assoc = pd.DataFrame(rows)
    assoc.to_csv(results / "score_association.csv", index=False)

    # ---------------- 6. pooled capture@20% with bootstrap CI --------------------------
    rows = []
    for m in methods:
        c = cubes[m]
        for label in ["failure", "relative_failure"]:
            for s in SCORES + ["condition-only:" + s for s in SCORES]:
                def _cap(cc):
                    vals = []
                    for si in range(len(seeds)):
                        lab = cc[label][:, si, :].reshape(-1).astype(bool)
                        if s.startswith("condition-only:"):
                            sc = cc[s.split(":", 1)[1]][:, si, :]
                            sc = np.broadcast_to(np.nanmean(sc, axis=0, keepdims=True), sc.shape)
                        else:
                            sc = cc[s][:, si, :]
                        vals.append(capture_rate(sc.reshape(-1), lab, 0.2))
                    vals = np.array(vals, float)
                    return np.nanmean(vals) if np.isfinite(vals).any() else np.nan
                est = _cap(c)
                bs = np.array([_cap({k: v[boot_idx[b]] for k, v in c.items()}) for b in range(min(B, 1000))], float)
                lo, hi = _ci(bs, ci)
                rows.append({"method": m, "label": label, "score": s, "fraction": 0.2, "capture": est, "ci_lo": lo, "ci_hi": hi})
    pd.DataFrame(rows).to_csv(results / "failure_capture_pooled_ci.csv", index=False)

    # ---------------- 7. representative cases (pre-defined rules) ---------------------
    cases = representative_cases(df, ratios)
    save_json(cases, results / "representative_cases.json")

    summary = {"n_test_images": int(n_img), "noise_seeds": [int(s) for s in seeds], "ratios": [float(r) for r in ratios],
               "bootstrap_resamples": B, "ci": ci}
    save_json(summary, results / "summary.json")
    log(f"[summarize] wrote summary tables to {results}")
    return summary


def representative_cases(df: pd.DataFrame, ratios) -> dict:
    """Rules fixed in advance (primary method: PnP-ADMM; noise seed 0):
    matched_median   : ratio 1.0, image with the median PSNR (lower middle for even n)
    mismatch_worst   : ratio != 1.0, largest MSE
    low_residual_high_error : among reconstructions whose relative residual is in the lowest
                       quartile (pooled over ratios), the largest-MSE one, reported only if it
                       is an absolute failure; otherwise 'none found'.
    """
    out = {}
    for m in df.method.unique():
        g = df[(df.method == m) & (df.noise_seed == df.noise_seed.min())]
        mm = g[np.isclose(g.ratio, 1.0)].sort_values(["psnr", "image_index"]).reset_index(drop=True)
        med = mm.iloc[(len(mm) - 1) // 2] if len(mm) else None
        mis = g[~np.isclose(g.ratio, 1.0)]
        worst = mis.loc[mis.mse.idxmax()] if len(mis) else None
        q = g.rel_residual.quantile(0.25)
        low = g[g.rel_residual <= q]
        cand = low.loc[low.mse.idxmax()] if len(low) else None

        def _row(r):
            if r is None:
                return None
            return {k: (v.item() if hasattr(v, "item") else v) for k, v in
                    r[["image_id", "image_index", "noise_seed", "ratio", "psnr", "mse", "rel_residual", "residual_acf",
                       "sensitivity", "failure"]].to_dict().items()}

        out[m] = {
            "matched_median": _row(med),
            "mismatch_worst": _row(worst),
            "low_residual_high_error": _row(cand) if cand is not None and bool(cand["failure"]) else "none found",
            "low_residual_quartile_threshold": float(q),
        }
    return out
