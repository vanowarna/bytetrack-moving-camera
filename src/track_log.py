"""Read-only instrumentation of BYTETracker's association steps (no change to ByteTrack's code or results).

`AssociationLogger.patch()` temporarily wraps `matching.iou_distance` and `matching.linear_assignment`
inside ByteTrack. Each BYTETracker.update calls them three times, in this order:
    stage 0  'high'        : all tracked + lost tracks (Kalman-predicted boxes) vs high-score detections
    stage 1  'low'         : still-unmatched tracked tracks vs low-score detections (BYTE's second step)
    stage 2  'unconfirmed' : new, unconfirmed tracks vs remaining high-score detections
For every track in a stage we record its *predicted* box (the Kalman prior, taken at the moment the IoU
is computed), whether it was matched, the IoU with the matched detection and the best IoU with any
detection. Tracker output is identical with and without the logger (checked in scripts/run_e3_log.py).
"""
from __future__ import annotations

import contextlib

import numpy as np

import src.bt_compat  # noqa: F401
from yolox.tracker import matching as _matching
from yolox.tracker.basetrack import TrackState

STAGES = ("high", "low", "unconfirmed")
COLUMNS = ["frame", "stage", "track_id", "state", "since", "px1", "py1", "px2", "py2", "vx", "vy",
           "matched", "dx1", "dy1", "dx2", "dy2", "det_score", "iou", "best_iou"]
_STATE = {TrackState.New: "new", TrackState.Tracked: "tracked", TrackState.Lost: "lost", TrackState.Removed: "removed"}


class AssociationLogger:
    def __init__(self):
        self.rows: list[list] = []
        self._frame = None
        self._tracker = None
        self._stage = 0
        self._pending = None

    def begin(self, frame: int, tracker):
        """Call right before tracker.update(...) for `frame`."""
        self._frame, self._tracker, self._stage, self._pending = frame, tracker, 0, None

    @contextlib.contextmanager
    def patch(self):
        orig_iou, orig_la = _matching.iou_distance, _matching.linear_assignment

        def iou_distance(atracks, btracks):
            cost = orig_iou(atracks, btracks)
            stage, self._stage = self._stage, self._stage + 1
            self._pending = None
            if self._frame is not None and stage < len(STAGES) and len(atracks) and not isinstance(atracks[0], np.ndarray):
                tr = self._tracker
                info = [(t.track_id, _STATE.get(t.state, str(t.state)), tr.frame_id - t.frame_id,
                         *np.asarray(t.tlbr, float), float(t.mean[4]), float(t.mean[5])) for t in atracks]
                dets = np.array([d.tlbr for d in btracks], float).reshape(-1, 4)
                scores = np.array([d.score for d in btracks], float)
                iou = 1.0 - cost if cost.size else np.zeros((len(atracks), len(btracks)))
                self._pending = (stage, info, dets, scores, np.asarray(iou, float))
            return cost

        def linear_assignment(cost_matrix, thresh):
            res = orig_la(cost_matrix, thresh)
            if self._pending is not None:
                self._record(res[0])
            self._pending = None
            return res

        _matching.iou_distance, _matching.linear_assignment = iou_distance, linear_assignment
        try:
            yield self
        finally:
            _matching.iou_distance, _matching.linear_assignment = orig_iou, orig_la

    def _record(self, matches):
        stage, info, dets, scores, iou = self._pending
        match = {int(i): int(j) for i, j in np.asarray(matches).reshape(-1, 2)}
        for i, (tid, state, since, x1, y1, x2, y2, vx, vy) in enumerate(info):
            best = float(iou[i].max()) if iou.shape[1] else 0.0
            if i in match:
                j = match[i]
                d = dets[j]
                row = [1, *d, scores[j], float(iou[i, j]), best]
            else:
                row = [0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, best]
            self.rows.append([self._frame, STAGES[stage], tid, state, since, x1, y1, x2, y2, vx, vy, *row])

    def to_frame(self):
        import pandas as pd
        return pd.DataFrame(self.rows, columns=COLUMNS)
