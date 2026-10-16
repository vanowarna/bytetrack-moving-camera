"""Record what produced a run: code commit, third-party commits, package versions, hardware."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from src.config import ROOT

PACKAGES = ["torch", "torchvision", "numpy", "opencv-python", "scipy", "lapx", "motmetrics", "pandas"]


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=cwd, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def environment(extra: dict | None = None) -> dict:
    env = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "repo_commit": _git(["rev-parse", "HEAD"], ROOT),
        "repo_dirty": bool(_git(["status", "--porcelain", "--untracked-files=no"], ROOT)),
        "bytetrack_commit": _git(["rev-parse", "HEAD"], ROOT / "external" / "ByteTrack"),
        "trackeval_commit": _git(["rev-parse", "HEAD"], ROOT / "external" / "TrackEval"),
        "packages": {},
    }
    for p in PACKAGES:
        try:
            env["packages"][p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            env["packages"][p] = None
    try:
        import torch
        env["cuda"] = torch.version.cuda
        env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        env["cuda"], env["gpu"] = None, None
    if extra:
        env.update(extra)
    return env


def write_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
