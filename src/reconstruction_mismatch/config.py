"""YAML configuration: base + profile (+ optional extra file) deep merge."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import yaml

from .utils.provenance import project_root


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and "logspace" not in v:
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _resolve(obj):
    if isinstance(obj, dict):
        if set(obj) == {"logspace"}:
            a, b, n = obj["logspace"]
            return [float(f"{v:.6g}") for v in np.logspace(float(a), float(b), int(n))]
        return {k: _resolve(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve(v) for v in obj]
    return obj


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(profile: str, extra=None, root: Path | None = None) -> dict:
    """extra: None, one path, or a list of paths merged in order."""
    root = Path(root) if root else project_root()
    cfg = load_yaml(root / "configs" / "base.yaml")
    prof_path = root / "configs" / "profiles" / f"{profile}.yaml"
    if not prof_path.exists():
        raise FileNotFoundError(f"unknown profile '{profile}' ({prof_path})")
    cfg = deep_merge(cfg, load_yaml(prof_path))
    extras = [extra] if isinstance(extra, (str, Path)) else list(extra or [])
    for e in extras:
        cfg = deep_merge(cfg, load_yaml(Path(e)))
    cfg = _resolve(cfg)
    cfg["profile"] = profile
    cfg["extra_config"] = [Path(e).as_posix() for e in extras]
    st = float(cfg["operator"]["sigma_true"])
    # The run stage only ever receives these assumed values, never sigma_true itself.
    cfg["operator"]["sigma_assumed"] = [round(r * st, 6) for r in cfg["operator"]["ratios"]]
    return cfg


def dump_yaml(cfg: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
