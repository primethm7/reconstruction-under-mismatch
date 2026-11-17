"""Provenance: code identity, environment, versions.

Every result directory stores this so a CSV can be traced back to the code, packages and
device that produced it. Cache keys use *compute-relevant* code hashes (data generation,
operators, reconstruction, denoisers, scores) so that editing plotting code does not force
an hour-long recomputation, while editing a solver does.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import torch

from .. import __version__

PACKAGE_DIR = Path(__file__).resolve().parents[1]

COMPUTE_SUBPACKAGES = ("data", "operators", "reconstruction", "denoisers", "metrics")


def project_root() -> Path:
    return PACKAGE_DIR.parents[1]


def code_hash(subpackages: tuple[str, ...] | None = None) -> str:
    """SHA-256 over the source files of the given subpackages (all if None)."""
    h = hashlib.sha256()
    if subpackages is None:
        files = sorted(PACKAGE_DIR.rglob("*.py"))
    else:
        files = []
        for sp in subpackages:
            files.extend(sorted((PACKAGE_DIR / sp).rglob("*.py")))
    for f in files:
        h.update(f.relative_to(PACKAGE_DIR).as_posix().encode())
        h.update(f.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def git_info() -> dict:
    root = project_root()
    info: dict = {"commit": None, "dirty": None}
    try:
        info["commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True
        ).stdout
        info["dirty"] = bool(status.strip())
    except Exception as exc:  # no git, or no commits yet
        info["error"] = repr(exc)
    return info


def package_versions() -> dict:
    out = {}
    for pkg in ["numpy", "scipy", "scikit-image", "matplotlib", "pandas", "PyYAML", "torch", "markdown"]:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def environment(device_log: dict | None = None) -> dict:
    return {
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "package_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch_num_threads": torch.get_num_threads(),
        "packages": package_versions(),
        "device": device_log,
        "git": git_info(),
        "code_sha256_all": code_hash(None),
        "code_sha256_compute": code_hash(COMPUTE_SUBPACKAGES),
    }
