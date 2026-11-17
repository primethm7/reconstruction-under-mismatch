"""Synthetic grey-scale phantoms in [0, 1] (NOT real cells, tissue or medical images).

Each phantom is drawn on a 2x supersampled grid and box-averaged to the target size
(anti-aliasing). Components, all with random parameters from a seeded numpy Generator:

1. smooth background: offset + a few low-frequency cosines (<= 2 cycles / image)
2. large ellipses: flat or smooth dome-shaped profiles, positive or negative contrast
3. small discs (radius 1.5-5 px)
4. clusters of close small objects (radius 0.8-2 px, centre spacing 3-6 px)
5. thin line segments (width 0.7-2 px)
6. circular arcs (width 0.7-1.5 px)
7. sinusoidal grating patches (3-28 cycles / image, i.e. periods down to ~4.6 px)
Finally the image is shifted to be non-negative, rescaled so its 99.5th percentile maps to a
random level in [0.75, 1.0], and clipped to [0, 1].
"""

from __future__ import annotations

import numpy as np

GENERATOR_VERSION = "synthetic_v1"


def _soft(d_minus_edge: np.ndarray, pix: float) -> np.ndarray:
    """Anti-aliased indicator of (d < edge): 1 inside, 0 outside, linear over one sub-pixel."""
    return np.clip(0.5 - d_minus_edge / pix, 0.0, 1.0)


def _segment_distance(yy, xx, p0, p1):
    d = p1 - p0
    L2 = float(d @ d) + 1e-12
    t = np.clip(((yy - p0[0]) * d[0] + (xx - p0[1]) * d[1]) / L2, 0.0, 1.0)
    py = p0[0] + t * d[0]
    px = p0[1] + t * d[1]
    return np.sqrt((yy - py) ** 2 + (xx - px) ** 2)


def _ellipse_r2(yy, xx, cy, cx, a, b, theta):
    c, s = np.cos(theta), np.sin(theta)
    dy, dx = yy - cy, xx - cx
    u = c * dx + s * dy
    v = -s * dx + c * dy
    return (u / a) ** 2 + (v / b) ** 2


def generate_phantom(rng: np.random.Generator, size: int = 128, supersample: int = 2) -> np.ndarray:
    n = size * supersample
    pix = 1.0 / supersample  # one sub-pixel, in final-pixel units
    c = (np.arange(n) + 0.5) * pix
    yy, xx = np.meshgrid(c, c, indexing="ij")
    img = np.zeros((n, n), dtype=np.float64)

    # 1. smooth background
    img += rng.uniform(0.05, 0.25)
    for _ in range(rng.integers(2, 5)):
        fy, fx = rng.uniform(-2.0, 2.0, size=2)
        img += rng.uniform(0.01, 0.06) * np.cos(2 * np.pi * (fy * yy + fx * xx) / size + rng.uniform(0, 2 * np.pi))

    # 2. large ellipses (flat or dome profile)
    for _ in range(rng.integers(2, 6)):
        cy, cx = rng.uniform(0.15, 0.85, size=2) * size
        a, b = rng.uniform(0.06, 0.25, size=2) * size
        r2 = _ellipse_r2(yy, xx, cy, cx, a, b, rng.uniform(0, np.pi))
        amp = rng.uniform(-0.2, 0.45)
        if rng.random() < 0.5:
            r = np.sqrt(r2)
            prof = _soft((r - 1.0) * min(a, b), pix)
        else:
            prof = np.clip(1.0 - r2, 0.0, None) ** rng.uniform(0.3, 1.5)
        img += amp * prof

    # 3. small discs
    for _ in range(rng.integers(3, 9)):
        cy, cx = rng.uniform(0.08, 0.92, size=2) * size
        r = rng.uniform(1.5, 5.0)
        img += rng.uniform(0.15, 0.5) * _soft(np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) - r, pix)

    # 4. clusters of close small objects
    for _ in range(rng.integers(1, 3)):
        cy, cx = rng.uniform(0.15, 0.85, size=2) * size
        k = int(rng.integers(3, 7))
        spacing = rng.uniform(3.0, 6.0)
        rad = rng.uniform(0.8, 2.0)
        amp = rng.uniform(0.25, 0.6)
        ang = rng.uniform(0, 2 * np.pi)
        for j in range(k):
            off = (j - (k - 1) / 2) * spacing
            oy = off * np.sin(ang) + rng.normal(0, 0.5)
            ox = off * np.cos(ang) + rng.normal(0, 0.5)
            img += amp * _soft(np.sqrt((yy - cy - oy) ** 2 + (xx - cx - ox) ** 2) - rad, pix)

    # 5. thin line segments
    for _ in range(rng.integers(2, 6)):
        p0 = rng.uniform(0.05, 0.95, size=2) * size
        p1 = p0 + rng.normal(0, 0.3, size=2) * size
        w = rng.uniform(0.7, 2.0)
        img += rng.uniform(0.15, 0.5) * _soft(_segment_distance(yy, xx, p0, p1) - w / 2, pix)

    # 6. circular arcs
    for _ in range(rng.integers(1, 3)):
        cy, cx = rng.uniform(0.2, 0.8, size=2) * size
        R = rng.uniform(0.08, 0.3) * size
        t0 = rng.uniform(0, 2 * np.pi)
        span = rng.uniform(0.5, 1.6) * np.pi
        w = rng.uniform(0.7, 1.5)
        ang = np.mod(np.arctan2(yy - cy, xx - cx) - t0, 2 * np.pi)
        on = ang < span
        d = np.abs(np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) - R)
        img += rng.uniform(0.15, 0.5) * _soft(d - w / 2, pix) * on

    # 7. grating patches inside soft ellipses
    for _ in range(rng.integers(1, 3)):
        cy, cx = rng.uniform(0.2, 0.8, size=2) * size
        a, b = rng.uniform(0.08, 0.2, size=2) * size
        r = np.sqrt(_ellipse_r2(yy, xx, cy, cx, a, b, rng.uniform(0, np.pi)))
        mask = _soft((r - 1.0) * min(a, b), pix)
        f = rng.uniform(3.0, 28.0)
        th = rng.uniform(0, np.pi)
        grating = np.cos(2 * np.pi * f * (np.cos(th) * xx + np.sin(th) * yy) / size + rng.uniform(0, 2 * np.pi))
        img += rng.uniform(0.05, 0.2) * grating * mask

    # global intensity normalisation
    img = img - img.min() + rng.uniform(0.0, 0.1)
    hi = np.percentile(img, 99.5)
    img = img / hi * rng.uniform(0.75, 1.0)
    img = np.clip(img, 0.0, 1.0)
    img = img.reshape(size, supersample, size, supersample).mean(axis=(1, 3))
    return img.astype(np.float32)
