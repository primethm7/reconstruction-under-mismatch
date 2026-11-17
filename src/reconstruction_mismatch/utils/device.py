"""Device selection: CUDA -> MPS -> CPU, with an actual compatibility check.

`torch.cuda.is_available()` alone is not enough (old drivers, broken installs), so each
candidate runs a tiny FFT round trip -- the operation every reconstruction here relies on.
"""

from __future__ import annotations

import torch


def _check(name: str) -> tuple[bool, str]:
    try:
        if name == "cuda":
            if not torch.cuda.is_available():
                return False, "torch.cuda.is_available() is False"
        elif name == "mps":
            mps = getattr(torch.backends, "mps", None)
            if mps is None or not mps.is_available():
                return False, "torch.backends.mps.is_available() is False"
        dev = torch.device(name)
        x = torch.arange(16, dtype=torch.float32, device=dev).reshape(1, 1, 4, 4)
        y = torch.fft.ifft2(torch.fft.fft2(x)).real
        err = float((y - x).abs().max().cpu())
        if err > 1e-4:
            return False, f"FFT round-trip error {err:.2e}"
        conv = torch.nn.Conv2d(1, 2, 3, padding=1).to(dev)
        conv(x)
        label = torch.cuda.get_device_name(0) if name == "cuda" else name
        return True, f"ok ({label})"
    except Exception as exc:  # pragma: no cover - depends on hardware
        return False, f"error: {exc!r}"


def select_device(preference: str = "auto") -> tuple[torch.device, dict]:
    """Return the first working device and a log of every check performed."""
    order = ["cuda", "mps", "cpu"] if preference == "auto" else [preference]
    log: dict = {"requested": preference, "checks": {}}
    for name in order:
        ok, msg = _check(name)
        log["checks"][name] = msg
        if ok:
            log["selected"] = name
            return torch.device(name), log
    raise RuntimeError(f"No usable device among {order}: {log}")
