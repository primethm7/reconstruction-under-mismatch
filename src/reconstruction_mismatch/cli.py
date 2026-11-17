"""Command-line interface.

    python -m reconstruction_mismatch <command> --profile smoke|quick|full [--config extra.yaml]

Commands:
    generate-data   synthetic (or MedMNIST, via --config) train/val/test splits
    train-denoiser  train (or reuse) the small CNN denoiser
    tune            hyper-parameter selection on validation data (freezes selection.json)
    run             test reconstructions + label-free scores (no ground truth access)
    evaluate        reference metrics + failure labels -> per_image.csv
    summarize       tables, failure-capture analysis, bootstrap intervals
    plot            figures
    report          fill reports/report_template.md with numbers from the CSVs
    reproduce       all of the above, in order (cached stages are reused)
"""

from __future__ import annotations

import argparse
import sys

from .config import load_config
from .experiments import pipeline as P


def _prepare(args):
    cfg = load_config(args.profile, args.config)
    ctx = P.Context(cfg).setup()
    ctx.info(f"profile={args.profile} tag={ctx.tag} device={ctx.device} ({ctx.device_log['checks']})")
    return ctx


def _through_denoiser(ctx):
    ddir, man = P.run_stage(ctx, "data", P.stage_data)
    ckpt = None
    if ctx.cfg["pnp"].get("denoiser", "small_cnn") == "small_cnn" or ctx.cfg.get("train_denoiser_anyway"):
        ckpt, _ = P.run_stage(ctx, "denoiser", P.stage_denoiser, ddir, man)
    return ddir, man, ckpt


def _through_tune(ctx):
    ddir, man, ckpt = _through_denoiser(ctx)
    den, den_info = P.build_denoiser(ctx, ckpt)
    if den_info["denoiser"] != "small_cnn":
        P.run_stage(ctx, "denoiser_eval", P.stage_eval_pretrained, den)
    val_obs = P.run_stage(ctx, "observations_val", P.stage_observations, ddir, man, "val")
    sel = P.run_stage(ctx, "tune", P.stage_tune, ddir, val_obs, den, den_info)
    return ddir, man, den, sel


def _through_run(ctx):
    ddir, man, den, sel = _through_tune(ctx)
    test_obs = P.run_stage(ctx, "observations_test", P.stage_observations, ddir, man, "test")
    run_dir = P.run_stage(ctx, "run", P.stage_run, test_obs, sel, den)
    return ddir, test_obs, run_dir, sel


def _through_evaluate(ctx):
    ddir, test_obs, run_dir, sel = _through_run(ctx)
    return P.run_stage(ctx, "evaluate", P.stage_evaluate, ddir, test_obs, run_dir, sel)


def cmd_summarize(ctx):
    from .experiments.analysis import summarize

    P.run_stage(ctx, "summarize", lambda c: summarize(c.results, c.cfg, log=c.info))


def cmd_plot(ctx):
    from .plotting.figures import make_all_figures

    P.run_stage(ctx, "plot", lambda c: make_all_figures(c.results, c.cfg, log=c.info))


def cmd_report(ctx):
    from .reporting.report import build_all

    P.run_stage(ctx, "report", lambda c: build_all(c.root, log=c.info))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="reconstruction_mismatch")
    ap.add_argument("command", choices=["generate-data", "train-denoiser", "tune", "run", "evaluate", "summarize",
                                        "plot", "report", "reproduce"])
    ap.add_argument("--profile", default="quick", help="smoke | quick | full")
    ap.add_argument("--config", action="append", default=[],
                    help="extra YAML merged on top, may be repeated (e.g. configs/extensions/...)")
    ap.add_argument("--no-report", action="store_true", help="reproduce: skip the report step")
    args = ap.parse_args(argv)
    ctx = _prepare(args)
    c = args.command
    if c == "generate-data":
        P.run_stage(ctx, "data", P.stage_data)
    elif c == "train-denoiser":
        _through_denoiser(ctx)
    elif c == "tune":
        _through_tune(ctx)
    elif c == "run":
        _through_run(ctx)
    elif c == "evaluate":
        _through_evaluate(ctx)
    elif c == "summarize":
        cmd_summarize(ctx)
    elif c == "plot":
        cmd_plot(ctx)
    elif c == "report":
        cmd_report(ctx)
    elif c == "reproduce":
        _through_evaluate(ctx)
        cmd_summarize(ctx)
        cmd_plot(ctx)
        if not args.no_report and ctx.tag == "full":
            cmd_report(ctx)
    ctx.info(f"[{c}] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
