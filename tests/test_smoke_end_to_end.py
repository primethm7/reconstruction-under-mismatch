"""End-to-end smoke run of the real CLI (slow: ~1-2 min; reuses cached stages)."""

import numpy as np
import pandas as pd
import pytest

from reconstruction_mismatch.cli import main
from reconstruction_mismatch.utils.io import load_json
from reconstruction_mismatch.utils.provenance import project_root


@pytest.mark.slow
def test_smoke_reproduce_runs_and_is_deterministic():
    res = project_root() / "results" / "smoke"
    assert main(["reproduce", "--profile", "smoke"]) == 0
    df1 = pd.read_csv(res / "per_image.csv")
    assert len(df1) == 4 * 1 * 5 * 2  # images x seeds x ratios x methods
    num = df1.select_dtypes("number")
    assert np.isfinite(num.drop(columns=["final_rel_change"])).all().all()
    assert not df1.diverged.any() and not df1.nonfinite_output.any()
    status = load_json(res / "run_status.json")
    assert all(v["status"] == "done" for v in status.values())
    for f in ["fig1_quality_vs_ratio.png", "fig2_scores_vs_error.png", "fig3_failure_capture.png", "fig4_examples.png"]:
        assert (res / "figures" / f).exists()
    env = load_json(res / "environment.json")
    assert env["code_sha256_all"] and env["device"]["selected"]
    # second invocation: cached stages, identical numbers
    assert main(["evaluate", "--profile", "smoke"]) == 0
    df2 = pd.read_csv(res / "per_image.csv")
    pd.testing.assert_frame_equal(df1, df2)
