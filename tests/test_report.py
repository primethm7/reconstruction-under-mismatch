"""Report placeholders must resolve to exactly one CSV row (no silent drift)."""

import pandas as pd
import pytest

from reconstruction_mismatch.reporting.report import _csv, _fmt, _json
from reconstruction_mismatch.utils.io import save_json


def test_csv_placeholder_exact_match(tmp_path):
    pd.DataFrame({"method": ["a", "a", "b"], "ratio": [1.0, 0.5, 1.0], "psnr": [30.123, 25.0, 28.0]}).to_csv(
        tmp_path / "t.csv", index=False)
    assert _csv(tmp_path, "t.csv|method=a,ratio=1.0|psnr|.2f") == "30.12"
    assert _csv(tmp_path, "t.csv|method=b,ratio=1|psnr|.1f") == "28.0"
    with pytest.raises(ValueError):
        _csv(tmp_path, "t.csv|method=a|psnr|.2f")  # two rows
    with pytest.raises(ValueError):
        _csv(tmp_path, "t.csv|method=c|psnr|.2f")  # zero rows


def test_json_placeholder_and_formats(tmp_path):
    save_json({"pnp": {"rho": 0.2, "ok": True}}, tmp_path / "s.json")
    assert _json(tmp_path, "s.json|pnp.rho|g") == "0.2"
    assert _json(tmp_path, "s.json|pnp.ok|") == "yes"
    assert _fmt(0.256, "pct") == "26%"
    assert _fmt(float("nan"), ".2f") == "n/a"
