"""Static figures (PNG) from the result CSVs.

Colour roles (validated with the dataviz palette checker, light surface):
  methods / scores : categorical slots blue #2a78d6, orange #eb6834, aqua #1baf7a (fixed order)
  mismatch ratio   : one-hue ordinal blue ramp (light -> dark = small -> large ratio)
  references       : grey dashed (random review), dark grey dotted (oracle)
Aqua is below 3:1 contrast on white, so every line also has a distinct marker and a
direct label; the numbers are in the CSV tables.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from ..utils.io import load_json, load_npz  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
CAT = ["#2a78d6", "#eb6834", "#1baf7a"]
MARKERS = ["o", "s", "^"]
RAMP5 = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
METHOD_LABEL = {"tikhonov": "Tikhonov", "pnp_admm": "PnP-ADMM"}
SCORE_LABEL = {"rel_residual": "relative residual", "residual_acf": "residual autocorr.", "sensitivity": "operator sensitivity"}
SCORES = list(SCORE_LABEL)


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
        "legend.frameon": False, "lines.linewidth": 1.6, "lines.markersize": 5.5,
    })


def _dodge(values: list[float], min_gap: float) -> list[float]:
    """Spread label y-positions so that consecutive labels are at least min_gap apart."""
    order = np.argsort(values)
    out = np.array(values, dtype=float)
    for a, b in zip(order[:-1], order[1:]):
        if out[b] - out[a] < min_gap:
            out[b] = out[a] + min_gap
    return out.tolist()


def _ramp(n: int) -> list[str]:
    return RAMP5 if n == 5 else [RAMP5[int(i)] for i in np.linspace(0, 4, n).round()]


def fig_quality_vs_ratio(results: Path, out: Path) -> Path:
    perf = pd.read_csv(results / "summary_by_condition.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))
    for ax, metric, lab in [(axes[0], "psnr", "PSNR (dB)"), (axes[1], "ssim", "SSIM")]:
        for i, m in enumerate([m for m in METHOD_LABEL if m in perf.method.unique()]):
            g = perf[perf.method == m].sort_values("ratio")
            ax.fill_between(g.ratio, g[f"{metric}_ci_lo"], g[f"{metric}_ci_hi"], color=CAT[i], alpha=0.15, linewidth=0)
            ax.plot(g.ratio, g[f"{metric}_mean"], color=CAT[i], marker=MARKERS[i], label=METHOD_LABEL[m])
            last = g.iloc[-1]
            ax.annotate(METHOD_LABEL[m], (last.ratio, last[f"{metric}_mean"]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK2)
        ax.axvline(1.0, color=INK2, linestyle="--", linewidth=0.8)
        ax.text(1.0, 0.02, " matched\n reference", transform=ax.get_xaxis_transform(), fontsize=7, color=INK2, va="bottom")
        ax.set_xlabel(r"$\sigma_{assumed} / \sigma_{true}$")
        ax.set_ylabel(lab)
        ax.set_xticks(sorted(perf.ratio.unique()))
        ax.set_xlim(perf.ratio.min() - 0.05, perf.ratio.max() + 0.22)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Reconstruction quality vs. assumed blur width (mean over test images, 95% cluster-bootstrap band)", fontsize=9)
    fig.tight_layout()
    p = out / "fig1_quality_vs_ratio.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def fig_scores_vs_error(results: Path, out: Path) -> Path:
    df = pd.read_csv(results / "per_image.csv")
    assoc = pd.read_csv(results / "score_association.csv")
    methods = [m for m in METHOD_LABEL if m in df.method.unique()]
    ratios = sorted(df.ratio.unique())
    cols = _ramp(len(ratios))
    fig, axes = plt.subplots(len(methods), 3, figsize=(11.0, 3.2 * len(methods)), squeeze=False)
    for r_i, m in enumerate(methods):
        g = df[df.method == m]
        for c_i, s in enumerate(SCORES):
            ax = axes[r_i, c_i]
            for k, r in enumerate(ratios):
                h = g[g.ratio == r]
                ax.scatter(h[s], h.mse, s=9, color=cols[k], alpha=0.75, linewidths=0, label=f"{r:g}")
            ax.set_xscale("log")
            ax.set_yscale("log")
            a = assoc[(assoc.method == m) & (assoc.score == s) & (assoc.scope == "pooled") & (assoc.statistic == "spearman_vs_mse")]
            w = assoc[(assoc.method == m) & (assoc.score == s) & (assoc.scope == "within") & (assoc.statistic == "spearman_vs_mse")]
            ttl = f"{METHOD_LABEL[m]}: {SCORE_LABEL[s]}"
            if len(a):
                ttl += f"\nSpearman: pooled {a.estimate.iloc[0]:+.2f} | within-ratio mean {w.estimate.mean():+.2f}"
            ax.set_title(ttl, fontsize=7.5)
            ax.set_xlabel(SCORE_LABEL[s] + " (label-free)")
            if c_i == 0:
                ax.set_ylabel("MSE vs. ground truth")
    axes[0, -1].legend(title=r"$\sigma_a/\sigma_t$", fontsize=7, title_fontsize=7, loc="upper left",
                       bbox_to_anchor=(1.0, 1.0), markerscale=1.6)
    fig.tight_layout()
    p = out / "fig2_scores_vs_error.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def fig_failure_capture(results: Path, out: Path) -> Path:
    pooled = pd.read_csv(results / "failure_capture_pooled.csv")
    within = pd.read_csv(results / "failure_capture_within.csv")
    methods = [m for m in METHOD_LABEL if m in pooled.method.unique()]
    fig, axes = plt.subplots(len(methods), 3, figsize=(10.2, 3.0 * len(methods)), squeeze=False)
    for r_i, m in enumerate(methods):
        panels = [
            ("pooled across ratios\nabsolute failure label", pooled[(pooled.method == m) & (pooled.label == "failure")], ""),
            ("pooled, CONDITION-ONLY scores\n(score replaced by its condition mean)",
             pooled[(pooled.method == m) & (pooled.label == "failure")], "[condition-only]"),
            ("within each ratio (mean over ratios)\nrelative label: worst 20% MSE", None, ""),
        ]
        for c_i, (title, data, suffix) in enumerate(panels):
            ax = axes[r_i, c_i]
            if data is None:
                w = within[(within.method == m) & (within.label == "relative_failure")]
                data = w.groupby(["score", "fraction"], as_index=False).capture.mean()
            ends = []
            for k, s in enumerate(SCORES):
                d = data[data.score == s + suffix].sort_values("fraction")
                if not len(d) or d.capture.isna().all():
                    continue
                ax.plot(d.fraction, d.capture, color=CAT[k], marker=MARKERS[k], label=SCORE_LABEL[s])
                mid = d[d.fraction <= 0.5].iloc[-1]
                ends.append((SCORE_LABEL[s].split()[-1], float(mid.fraction), float(mid.capture)))
            for (txt, xx, yy), ylab in zip(ends, _dodge([e[2] for e in ends], 0.075)):
                ax.annotate(txt, (xx, yy), xytext=(xx + 0.03, ylab - 0.035), fontsize=7, color=INK2)
            for ref, ls, col in [("random", "--", "#8a8984"), ("oracle", ":", INK2)]:
                d = data[data.score == ref].sort_values("fraction")
                if len(d) and not d.capture.isna().all():
                    ax.plot(d.fraction, d.capture, color=col, linestyle=ls, linewidth=1.2, label=f"{ref} ({'expected' if ref == 'random' else 'true MSE rank'})")
            if not ends:
                ax.text(0.5, 0.5, "undefined:\nno failures in this set", transform=ax.transAxes, ha="center",
                        va="center", fontsize=8, color=INK2)
            ax.set_ylim(-0.02, 1.04)
            ax.set_xlim(0, 1.02)
            ax.set_title(f"{METHOD_LABEL[m]}: {title}", fontsize=8)
            ax.set_xlabel("review fraction")
            if c_i == 0:
                ax.set_ylabel("failure capture rate")
    handles, labels = [], []
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    p = out / "fig3_failure_capture.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def fig_examples(results: Path, out: Path, root: Path) -> Path | None:
    from ..data.observations import load_observations, load_simulation_meta
    from ..data.splits import load_split_images
    from ..operators import GaussianBlur

    cases = load_json(results / "representative_cases.json")
    man = load_json(results / "run_manifest.json")
    run_dir = root / man["run_dir"]
    sim = load_json(results / "simulation_test.json")
    obs_dir = root / sim["directory"]
    obs = load_observations(obs_dir)
    st = load_simulation_meta(obs_dir)["sigma_true"]
    ddir = root / load_json(results / "data_manifest.json")["directory"]
    gt = load_split_images(ddir, "test")
    primary = "pnp_admm" if "pnp_admm" in cases else next(iter(cases))
    sel = []
    for name, label in [("matched_median", "matched: median-PSNR case"), ("mismatch_worst", "mismatch: largest MSE"),
                        ("low_residual_high_error", "low residual but failure")]:
        c = cases[primary][name]
        if isinstance(c, dict):
            sel.append((label, c))
    methods = [m for m in METHOD_LABEL if m in cases]
    ncol = 3 + 2 * len(methods)
    fig, axes = plt.subplots(len(sel), ncol, figsize=(1.75 * ncol, 1.95 * len(sel) + 0.3), squeeze=False)
    for r_i, (label, c) in enumerate(sel):
        i_obs = int(np.where((obs.image_index == c["image_index"]) & (obs.noise_seed == c["noise_seed"]))[0][0])
        sa = round(float(c["ratio"]) * st, 6)
        x_true = gt[c["image_index"]]
        y = obs.y[i_obs]
        panels = [("ground truth", x_true, "gray", (0, 1)), (f"observation y", y, "gray", (0, 1))]
        recs = {}
        for m in methods:
            d = load_npz(run_dir / f"{m}__sa{sa:.4f}.npz")
            recs[m] = d["x_hat"][i_obs]
            from ..metrics.image import psnr
            panels.append((f"{METHOD_LABEL[m]}\nPSNR {psnr(recs[m], x_true):.1f} dB", recs[m], "gray", (0, 1)))
        err_max = max(np.abs(recs[m] - x_true).max() for m in methods)
        for m in methods:
            panels.append((f"|error| {METHOD_LABEL[m]}", np.abs(recs[m] - x_true), "Oranges", (0, err_max)))
        op = GaussianBlur(sa, x_true.shape, truncate=4.0)
        res = (op(torch.from_numpy(recs[primary])[None, None]) - torch.from_numpy(y)[None, None])[0, 0].numpy()
        rmax = np.abs(res).max()
        panels.append((f"residual A_a x - y\n({METHOD_LABEL[primary]})", res, "RdBu_r", (-rmax, rmax)))
        for c_i, (t, img, cmap, (lo, hi)) in enumerate(panels):
            ax = axes[r_i, c_i]
            ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            ax.set_title(t, fontsize=7)
        axes[r_i, 0].set_ylabel(f"{label}\n{c['image_id']} seed {c['noise_seed']}\nratio {c['ratio']:g}", fontsize=7)
    fig.tight_layout()
    p = out / "fig4_examples.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_denoiser(results: Path, out: Path) -> Path | None:
    f = results / "denoiser" / "train_log.csv"
    if not f.exists():
        return None
    h = pd.read_csv(f)
    sig = sorted(h.sigma.unique())
    cols = _ramp(len(sig))
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0))
    loss = h.dropna(subset=["train_loss"]).drop_duplicates("iteration")
    axes[0].plot(loss.iteration, loss.train_loss, color=CAT[0], marker="o", markersize=3)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("training iteration")
    axes[0].set_ylabel("training MSE (running mean)")
    axes[0].set_title("Training loss (synthetic train split)", fontsize=9)
    for k, s in enumerate(sig):
        g = h[h.sigma == s].sort_values("iteration")
        axes[1].plot(g.iteration, g.psnr_denoised - g.psnr_noisy, color=cols[k], marker="o", markersize=3, label=f"σ={s:g}")
        axes[1].annotate(f"σ={s:g}", (g.iteration.iloc[-1], (g.psnr_denoised - g.psnr_noisy).iloc[-1]), xytext=(4, 0),
                         textcoords="offset points", fontsize=7, color=INK2, va="center")
    axes[1].axhline(0, color=INK2, linewidth=0.8)
    axes[1].set_xlabel("training iteration")
    axes[1].set_ylabel("PSNR gain over noisy input (dB)")
    axes[1].set_title("Held-out denoising gain (validation phantoms)", fontsize=9)
    axes[1].set_ylim(bottom=max(-3, axes[1].get_ylim()[0]))
    axes[1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    p = out / "fig5_denoiser_training.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def fig_tuning(results: Path, out: Path) -> Path | None:
    t = results / "tuning"
    if not (t / "tuning_tikhonov.csv").exists():
        return None
    tik = pd.read_csv(t / "tuning_tikhonov.csv")
    pnp = pd.read_csv(t / "tuning_pnp.csv")
    sel = load_json(t / "selection.json")
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))
    axes[0].plot(tik["lambda"], tik.psnr_mean, color=CAT[0], marker="o", markersize=3)
    axes[0].axvline(sel["tikhonov"]["lam"], color=INK2, linestyle="--", linewidth=0.8)
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"$\lambda$")
    axes[0].set_ylabel("validation PSNR (dB)")
    axes[0].set_title(f"Tikhonov (selected λ={sel['tikhonov']['lam']:.3g})", fontsize=9)
    K = sel["pnp_admm"]["n_iter"]
    piv = pnp[pnp.n_iter == K].pivot(index="sigma_d", columns="rho", values="psnr_mean")
    imh = axes[1].imshow(piv.values, cmap="Blues", aspect="auto", origin="lower")
    axes[1].set_xticks(range(len(piv.columns)), [f"{c:g}" for c in piv.columns])
    axes[1].set_yticks(range(len(piv.index)), [f"{c:g}" for c in piv.index])
    axes[1].set_xlabel(r"$\rho$")
    axes[1].set_ylabel(r"$\sigma_d$")
    axes[1].grid(False)
    for (a, b), v in np.ndenumerate(piv.values):
        axes[1].text(b, a, f"{v:.1f}", ha="center", va="center", fontsize=6,
                     color="white" if v > np.nanpercentile(piv.values, 60) else INK)
    axes[1].set_title(f"PnP-ADMM validation PSNR at K={K}", fontsize=9)
    fig.colorbar(imh, ax=axes[1], fraction=0.046)
    best = pnp[(pnp.rho == sel["pnp_admm"]["rho"]) & (pnp.sigma_d == sel["pnp_admm"]["sigma_d"])].sort_values("n_iter")
    axes[2].plot(best.n_iter, best.psnr_mean, color=CAT[1], marker="s", markersize=4)
    axes[2].set_xlabel("iterations K")
    axes[2].set_ylabel("validation PSNR (dB)")
    axes[2].set_title(f"PnP-ADMM vs K (ρ={sel['pnp_admm']['rho']:g}, σ_d={sel['pnp_admm']['sigma_d']:g})", fontsize=9)
    fig.tight_layout()
    p = out / "fig6_tuning.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def make_all_figures(results: Path, cfg: dict, log=print) -> list[Path]:
    from ..utils.provenance import project_root

    _style()
    results = Path(results)
    out = results / "figures"
    out.mkdir(parents=True, exist_ok=True)
    made = []
    for fn in [fig_quality_vs_ratio, fig_scores_vs_error, fig_failure_capture, fig_denoiser, fig_tuning]:
        p = fn(results, out)
        if p:
            made.append(p)
    p = fig_examples(results, out, project_root())
    if p:
        made.append(p)
    log(f"[plot] {len(made)} figures -> {out}")
    return made
