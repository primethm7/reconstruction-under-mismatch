"""Failure-capture ("review priority") metrics.

Setting: n reconstructions, a binary failure label per reconstruction (from ground truth;
evaluation only) and a label-free score (higher = review first). A reviewer inspects the
top-k by score, k = ceil(f * n) for review fraction f (so k >= 1 and k/n >= f; the
effective fraction k/n is reported).

* capture(f) = (# failures among the top-k) / (# failures). Undefined (NaN) if there are
  no failures.
* Ties: the boundary group of tied scores is split proportionally, i.e. the expected
  capture under uniformly random tie-breaking. NaN scores are ranked first (treated as most
  suspicious) -- a reconstruction whose score cannot be computed should be reviewed.
* random(f) = k / n: expected capture of a uniformly random review order.
* oracle(f) = min(k, F) / F: ranking by the true error (evaluation-only reference).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import rankdata, spearmanr


def review_k(n: int, fraction: float) -> int:
    return max(1, min(n, int(math.ceil(fraction * n - 1e-9))))


def expected_captured(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    s = np.where(np.isnan(scores), np.inf, np.asarray(scores, dtype=np.float64))
    lab = np.asarray(labels, dtype=np.float64)
    thr = np.sort(s)[::-1][k - 1]
    above = s > thr
    tie = s == thr
    return float(lab[above].sum() + (k - above.sum()) * lab[tie].sum() / tie.sum())


def capture_rate(scores, labels, fraction: float) -> float:
    labels = np.asarray(labels, dtype=bool)
    n_fail = labels.sum()
    if n_fail == 0:
        return float("nan")
    return expected_captured(np.asarray(scores), labels, review_k(len(labels), fraction)) / n_fail


def random_capture(n: int, fraction: float) -> float:
    return review_k(n, fraction) / n


def oracle_capture(labels, fraction: float) -> float:
    labels = np.asarray(labels, dtype=bool)
    n_fail = labels.sum()
    if n_fail == 0:
        return float("nan")
    return min(review_k(len(labels), fraction), n_fail) / n_fail


def auroc(scores, labels) -> float:
    """Probability that a random failure scores higher than a random non-failure (ties 1/2)."""
    labels = np.asarray(labels, dtype=bool)
    s = np.where(np.isnan(scores), np.inf, np.asarray(scores, dtype=np.float64))
    n1, n0 = labels.sum(), (~labels).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    return float(spearmanr(a, b).statistic)
