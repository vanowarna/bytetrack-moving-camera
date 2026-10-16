"""Detection cache I/O (no torch dependency)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def save_dets(path: Path, frames: list[int], dets: list[np.ndarray | None], ms: list[float], hw: tuple[int, int]):
    """Store all frames of one sequence in a single .npz. Rows: [frame, x1, y1, x2, y2, score].
    `none_frames` lists frames where postprocess returned None (tracker is not updated there)."""
    rows, none_frames = [], []
    for f, d in zip(frames, dets):
        if d is None:
            none_frames.append(f)
            continue
        rows.append(np.concatenate([np.full((len(d), 1), f, np.float32), d], axis=1))
    arr = np.concatenate(rows, axis=0) if rows else np.zeros((0, 6), np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, dets=arr, frames=np.asarray(frames, np.int32),
                        none_frames=np.asarray(none_frames, np.int32), ms=np.asarray(ms, np.float32),
                        hw=np.asarray(hw, np.int32))


def load_dets(path: Path) -> dict:
    """Return {'by_frame': {frame: (N,5) or None}, 'frames': [...], 'hw': (h, w), 'ms': array}."""
    z = np.load(path)
    dets, frames = z["dets"], z["frames"].tolist()
    none = set(z["none_frames"].tolist())
    by_frame: dict[int, np.ndarray | None] = {f: (None if f in none else np.zeros((0, 5), np.float32)) for f in frames}
    if len(dets):
        order = np.argsort(dets[:, 0], kind="stable")
        dets = dets[order]
        fr = dets[:, 0].astype(int)
        splits = np.flatnonzero(np.diff(fr)) + 1
        for chunk in np.split(dets, splits):
            by_frame[int(chunk[0, 0])] = chunk[:, 1:].copy()
    return {"by_frame": by_frame, "frames": frames, "hw": tuple(z["hw"].tolist()), "ms": z["ms"]}
