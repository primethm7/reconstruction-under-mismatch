"""Fill reports/report_template.md with numbers read from result files; render HTML/PDF.

Placeholders (every number in the report comes from a result file, never typed by hand):
  {{csv:PATH|col=val,col2=val2|COLUMN|FMT}}  value from the single CSV row matching the filters
  {{json:PATH|dotted.key|FMT}}               value from a JSON file
  {{table:NAME}}                             generated Markdown table (see TABLES)
  {{fig:FILE|caption}}                       copies results/<FILE> into reports/figures/
PATH is relative to results/. Numeric filter values match with a 1e-9 tolerance. A
placeholder that matches zero or several rows raises an error, so the report cannot drift
from the CSVs silently.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import markdown
import numpy as np
import pandas as pd

from ..utils.io import load_json, write_text

PAT = re.compile(r"\{\{(csv|json|table|fig):([^}]*)\}\}")
METHOD = {"tikhonov": "Tikhonov", "pnp_admm": "PnP-ADMM"}
SCORE = {"rel_residual": "relative residual", "residual_acf": "residual autocorrelation", "sensitivity": "operator sensitivity"}


def _fmt(v, fmt: str) -> str:
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    if fmt.startswith("abs:"):
        return _fmt(abs(float(v)), fmt[4:])
    if fmt == "pct":
        return f"{100 * float(v):.0f}%"
    if fmt in ("", "s"):
        return str(v)
    return format(float(v), fmt) if fmt not in ("d",) else str(int(v))


def _match(df: pd.DataFrame, filters: str) -> pd.DataFrame:
    m = pd.Series(True, index=df.index)
    for part in [p for p in filters.split(",") if p.strip()]:
        k, v = part.split("=", 1)
        col = df[k.strip()]
        v = v.strip()
        try:
            fv = float(v)
            num = pd.to_numeric(col, errors="coerce")
            m &= (num - fv).abs() < 1e-9
        except ValueError:
            m &= col.astype(str) == v
    return df[m]


def _csv(results: Path, arg: str) -> str:
    path, filters, column, fmt = (arg.split("|") + ["", "", ""])[:4]
    df = pd.read_csv(results / path)
    rows = _match(df, filters)
    if len(rows) != 1:
        raise ValueError(f"{{{{csv:{arg}}}}} matched {len(rows)} rows")
    return _fmt(rows.iloc[0][column], fmt)


def _json(results: Path, arg: str) -> str:
    path, key, fmt = (arg.split("|") + ["", ""])[:3]
    obj = load_json(results / path)
    for k in key.split("."):
        obj = obj[int(k)] if isinstance(obj, list) else obj[k]
    return _fmt(obj, fmt)


# ------------------------------- tables ------------------------------------------------


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def table_perf(results: Path, tag: str = "full") -> str:
    p = pd.read_csv(results / tag / "summary_by_condition.csv")
    rows = []
    for _, r in p.sort_values(["method", "ratio"], key=lambda s: s.map({"tikhonov": 0, "pnp_admm": 1}) if s.name == "method" else s).iterrows():
        rows.append({
            "method": METHOD.get(r.method, r.method),
            "σa/σt": f"{r.ratio:g}" + (" (matched ref.)" if abs(r.ratio - 1) < 1e-9 else ""),
            "PSNR dB [95% CI]": f"{r.psnr_mean:.2f} [{r.psnr_ci_lo:.2f}, {r.psnr_ci_hi:.2f}]",
            "SSIM [95% CI]": f"{r.ssim_mean:.3f} [{r.ssim_ci_lo:.3f}, {r.ssim_ci_hi:.3f}]",
            "abs. failures": f"{int(r.n_failures)}/{int(r.n_obs)}",
            "runtime / image": f"{1000 * r.runtime_s_per_image:.1f} ms",
        })
    return _md(pd.DataFrame(rows))


def table_runtime(results: Path, tag: str = "full") -> str:
    p = pd.read_csv(results / tag / "summary_by_condition.csv")
    g = p.groupby("method").agg(runtime=("runtime_s_per_image", "mean"), sens=("sens_runtime_s_per_image", "mean"),
                                iters=("iterations", "mean"), ndiv=("n_diverged", "sum"), nonfin=("n_nonfinite", "sum"),
                                nobs=("n_obs", "sum"), relchg=("final_rel_change_median", "median")).reset_index()
    rows = [{"method": METHOD.get(r.method, r.method), "reconstruction / image": f"{1000 * r.runtime:.1f} ms",
             "sensitivity score (2 extra reconstructions) / image": f"{1000 * r.sens:.1f} ms",
             "iterations": f"{r.iters:g}", "median final rel. change": "n/a (closed form)" if np.isnan(r.relchg) else f"{r.relchg:.1e}",
             "diverged / non-finite": f"{int(r.ndiv)} / {int(r.nonfin)} of {int(r.nobs)}"} for _, r in g.iterrows()]
    return _md(pd.DataFrame(rows))


def table_denoiser(results: Path, tag: str = "full") -> str:
    h = pd.read_csv(results / tag / "denoiser" / "heldout.csv")
    rows = [{"noise σ": f"{r.sigma:g}", "noisy input PSNR": f"{r.psnr_noisy:.2f}", "random-init model": f"{r.psnr_random_init:.2f}",
             "trained model": f"{r.psnr_trained:.2f}", "gain vs input": f"{r.gain_trained_db:+.2f} dB"} for _, r in h.iterrows()]
    return _md(pd.DataFrame(rows))


def table_assoc(results: Path, tag: str = "full") -> str:
    a = pd.read_csv(results / tag / "score_association.csv")
    rows = []
    for m in ["tikhonov", "pnp_admm"]:
        for s in SCORE:
            sub = a[(a.method == m) & (a.score == s)]
            if not len(sub):
                continue
            pooled = sub[(sub.scope == "pooled") & (sub.statistic == "spearman_vs_mse")].iloc[0]
            pa = sub[(sub.scope == "pooled") & (sub.statistic == "auroc_failure")].iloc[0]
            w = sub[(sub.scope == "within") & (sub.statistic == "spearman_vs_mse")]
            wm = w[np.isclose(pd.to_numeric(w.ratio, errors="coerce"), 1.0)].iloc[0]
            rows.append({"method": METHOD[m], "score": SCORE[s],
                         "Spearman vs MSE, pooled [95% CI]": f"{pooled.estimate:+.2f} [{pooled.ci_lo:+.2f}, {pooled.ci_hi:+.2f}]",
                         "AUROC failure, pooled [95% CI]": f"{pa.estimate:.2f} [{pa.ci_lo:.2f}, {pa.ci_hi:.2f}]",
                         "Spearman within ratio 1.0 [95% CI]": f"{wm.estimate:+.2f} [{wm.ci_lo:+.2f}, {wm.ci_hi:+.2f}]",
                         "Spearman within, range over ratios": f"{w.estimate.min():+.2f} to {w.estimate.max():+.2f}"})
    return _md(pd.DataFrame(rows))


def table_capture(results: Path, tag: str = "full") -> str:
    pc = pd.read_csv(results / tag / "failure_capture_pooled.csv")
    rows = []
    for m in ["tikhonov", "pnp_admm"]:
        for label in ["failure"]:
            sub = pc[(pc.method == m) & (pc.label == label)]
            if not len(sub):
                continue
            for s in list(SCORE) + ["random", "oracle"]:
                r = {"method": METHOD[m], "ranking": SCORE.get(s, s + (" (expected)" if s == "random" else " (true MSE)"))}
                for f in [0.1, 0.2, 0.3, 0.5]:
                    v = sub[(sub.score == s) & np.isclose(sub.fraction, f)].capture
                    r[f"{int(f * 100)}%"] = "n/a" if not len(v) or np.isnan(v.iloc[0]) else f"{v.iloc[0]:.2f}"
                    if s in SCORE:
                        vc = sub[(sub.score == f"{s}[condition-only]") & np.isclose(sub.fraction, f)].capture
                        r[f"{int(f * 100)}%"] += "" if not len(vc) or np.isnan(vc.iloc[0]) else f" ({vc.iloc[0]:.2f})"
                rows.append(r)
    return _md(pd.DataFrame(rows))


def table_capture_within(results: Path, tag: str = "full") -> str:
    w = pd.read_csv(results / tag / "failure_capture_within.csv")
    rows = []
    for m in ["tikhonov", "pnp_admm"]:
        sub = w[(w.method == m) & (w.label == "relative_failure") & np.isclose(w.fraction, 0.2)]
        if not len(sub):
            continue
        for s in list(SCORE) + ["random", "oracle"]:
            r = {"method": METHOD[m], "ranking": SCORE.get(s, s)}
            for ratio in sorted(sub.ratio.unique()):
                v = sub[(sub.score == s) & np.isclose(sub.ratio, ratio)].capture
                r[f"σa/σt={ratio:g}"] = "n/a" if not len(v) or np.isnan(v.iloc[0]) else f"{v.iloc[0]:.2f}"
            rows.append(r)
    return _md(pd.DataFrame(rows))


def table_abs_failures_within(results: Path, tag: str = "full") -> str:
    w = pd.read_csv(results / tag / "failure_capture_within.csv")
    sub = w[(w.label == "failure") & (w.score == "rel_residual") & np.isclose(w.fraction, 0.2)]
    rows = [{"method": METHOD.get(r.method, r.method), "σa/σt": f"{r.ratio:g}", "absolute failures per noise seed": r.failures_per_seed}
            for _, r in sub.iterrows()]
    return _md(pd.DataFrame(rows))


def table_extension(results: Path, tag: str) -> str:
    if not (results / tag / "summary_by_condition.csv").exists():
        return f"*(extension `{tag}` not run)*"
    p = pd.read_csv(results / tag / "summary_by_condition.csv")
    rows = []
    for m in ["tikhonov", "pnp_admm"]:
        g = p[p.method == m].sort_values("ratio")
        if not len(g):
            continue
        r = {"PSNR dB [95% CI]": METHOD[m]}
        for _, x in g.iterrows():
            r[f"σa/σt={x.ratio:g}"] = f"{x.psnr_mean:.2f} [{x.psnr_ci_lo:.1f}, {x.psnr_ci_hi:.1f}]"
        rows.append(r)
    return _md(pd.DataFrame(rows))


TABLES = {
    "perf": lambda R: table_perf(R),
    "runtime": lambda R: table_runtime(R),
    "denoiser": lambda R: table_denoiser(R),
    "assoc": lambda R: table_assoc(R),
    "capture_pooled": lambda R: table_capture(R),
    "capture_within": lambda R: table_capture_within(R),
    "abs_failures_within": lambda R: table_abs_failures_within(R),
}


def _table(results: Path, arg: str) -> str:
    name, _, tag = arg.partition("|")
    if name == "extension":
        return table_extension(results, tag)
    if name == "assoc_ext":
        return table_assoc(results, tag)
    if name == "capture_ext":
        return table_capture(results, tag)
    return TABLES[name](results)


# ------------------------------- rendering ---------------------------------------------

CSS = """
@page { size: A4; margin: 12mm 13mm; }
body { font-family: "Segoe UI", Arial, sans-serif; font-size: 9.2pt; line-height: 1.3; color: #0b0b0b;
       background: #ffffff; max-width: 190mm; margin: 0 auto; }
h1 { font-size: 15pt; margin: 0 0 3pt; } h2 { font-size: 11pt; margin: 9pt 0 3pt; border-bottom: 1px solid #ddd; }
h3 { font-size: 10.5pt; margin: 10pt 0 3pt; }
table { border-collapse: collapse; font-size: 7.4pt; margin: 4pt 0; width: 100%; } p { margin: 4pt 0; }
th, td { border: 1px solid #d0cfca; padding: 1pt 3pt; text-align: left; vertical-align: top; }
th { background: #f2f1ee; }
img { max-width: 88%; display: block; margin: 2pt auto; } img[alt^="Figure 1"] { max-width: 68%; }
figure { margin: 6pt 0; page-break-inside: avoid; } figcaption { font-size: 8.5pt; color: #52514e; }
code { font-family: Consolas, monospace; font-size: 8.5pt; background: #f4f4f2; padding: 0 2px; }
pre { background: #f4f4f2; padding: 5pt; font-size: 7.6pt; white-space: pre-wrap; word-break: break-word; }
blockquote { border-left: 3px solid #2a78d6; margin: 6pt 0; padding: 2pt 8pt; color: #333; }
"""


def find_browser() -> str | None:
    for p in [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Google\Chrome\Application\chrome.exe",
              "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "/usr/bin/chromium", "/usr/bin/google-chrome"]:
        if Path(p).exists():
            return p
    return None


def render(md_path: Path, log=print) -> dict:
    md_text = md_path.read_text(encoding="utf-8")
    body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "sane_lists"])
    html = f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>Reconstruction under Mismatch</title>" \
           f"<style>{CSS}</style></head><body>{body}</body></html>"
    html_path = md_path.with_suffix(".html")
    write_text(html_path, html)
    out = {"html": html_path, "pdf": None}
    browser = find_browser()
    if browser is None:
        log("[report] no Chromium-based browser found: HTML + Markdown only (no PDF)")
        return out
    pdf = md_path.with_suffix(".pdf")
    with tempfile.TemporaryDirectory() as prof:
        cmd = [browser, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--user-data-dir={prof}",
               f"--print-to-pdf={pdf}", html_path.resolve().as_uri()]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if pdf.exists() and pdf.stat().st_size > 0:
        out["pdf"] = pdf
        log(f"[report] PDF written with {Path(browser).name}")
    else:
        log(f"[report] PDF rendering failed ({r.returncode}): {r.stderr[-300:]}")
    return out


def build_report(root: Path, log=print, template: str = "report_template.md", output: str = "reports/report.md",
                 render_outputs: bool = True) -> dict:
    root = Path(root)
    results = root / "results"
    rep = root / "reports"
    figdir = rep / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    text = (rep / template).read_text(encoding="utf-8")
    md_path = root / output

    def sub(m: re.Match) -> str:
        kind, arg = m.group(1), m.group(2)
        if kind == "csv":
            return _csv(results, arg)
        if kind == "json":
            return _json(results, arg)
        if kind == "table":
            return _table(results, arg)
        if kind == "fig":
            f, _, cap = arg.partition("|")
            src = results / f
            dst = figdir / (f.replace("/figures/", "_").replace("/", "_"))
            shutil.copy2(src, dst)
            rel = Path(os.path.relpath(dst, md_path.parent)).as_posix()
            return f"<figure><img src=\"{rel}\" alt=\"{cap}\"><figcaption>{cap}</figcaption></figure>"
        raise ValueError(kind)

    filled = PAT.sub(sub, text)
    if "{{" in filled:
        raise ValueError("unfilled placeholder left in report")
    write_text(md_path, filled)
    out = render(md_path, log=log) if render_outputs else {"md": md_path}
    log(f"[report] {md_path} ({len(filled.split())} words)")
    return out


def build_all(root: Path, log=print) -> dict:
    """reports/report.{md,html,pdf} and README.md, both from templates in reports/.

    Any other ``reports/<NAME>_template.md`` is rendered to ``<NAME>.md`` in the repo root
    (markdown only), so local notes can use the same CSV placeholders.
    """
    out = build_report(root, log=log)
    reports = Path(root) / "reports"
    if (reports / "README_template.md").exists():
        out["readme"] = build_report(root, log=log, template="README_template.md", output="README.md",
                                     render_outputs=False)["md"]
    for tpl in sorted(reports.glob("*_template.md")):
        if tpl.name in ("report_template.md", "README_template.md"):
            continue
        dst = tpl.name[: -len("_template.md")] + ".md"
        out[dst] = build_report(root, log=log, template=tpl.name, output=dst, render_outputs=False)["md"]
    return out
