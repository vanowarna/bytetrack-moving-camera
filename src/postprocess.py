"""Offline post-processing of tracker output.

`interpolate` re-implements ByteTrack's linear tracklet interpolation (`tools/interpolation.py`, function
`dti`, MIT licence) on in-memory rows, because the original reads/writes files with POSIX-only path
handling. Behaviour is the same:
- only tracklets with more than `n_min` boxes are interpolated
- a gap between two consecutive boxes of the same track is filled when 1 < gap < `n_dti` frames
- filled boxes are linear in (x1, y1, w, h) and get score 1
The paper's "interval 20" setting is n_dti=20 (MOTA 76.6 -> 78.3, IDF1 79.3 -> 80.2 on MOT17 half-val).
"""
from __future__ import annotations

import numpy as np


def interpolate(rows: list[tuple], n_min: int = 25, n_dti: int = 20) -> list[tuple]:
    """rows: (frame, id, x1, y1, w, h, score). Returns rows with interpolated boxes added, sorted by frame."""
    if not rows:
        return rows
    data = np.asarray(rows, dtype=np.float64)
    out = [data]
    for tid in np.unique(data[:, 1]):
        tr = data[data[:, 1] == tid]
        tr = tr[np.argsort(tr[:, 0], kind="stable")]
        if len(tr) <= n_min:
            continue
        added = []
        for i in range(1, len(tr)):
            left, right = tr[i - 1], tr[i]
            gap = right[0] - left[0]
            if 1 < gap < n_dti:
                for j in range(1, int(gap)):
                    box = left[2:6] + (right[2:6] - left[2:6]) * j / gap
                    added.append([left[0] + j, tid, *box, 1.0])
        if added:
            out.append(np.asarray(added))
    allrows = np.concatenate(out, axis=0)
    allrows = allrows[np.argsort(allrows[:, 0], kind="stable")]
    return [(int(r[0]), int(r[1]), *r[2:6], r[6]) for r in allrows]
