"""Run the official BYTETracker on cached detections, reproducing ByteTrack's evaluator.

Faithful to `yolox/evaluators/mot_evaluator.py` (MOTEvaluator.evaluate):
- a new BYTETracker(args, frame_rate=30) per sequence
- per-sequence track_buffer override (MOT17-05 -> 14, MOT17-13 -> 25, else 30) when enabled
- tracker.update is NOT called on frames where the detector returned None
- output filter: keep a track box if w*h > min_box_area and w/h <= aspect_ratio_thresh
- result line: frame,id,x1,y1,w,h,score,-1,-1,-1 (box rounded to 0.1, score to 0.01)
Only addition: an optional Kalman noise scale for the sensitivity study (default 1.0 = official).

`tracker.type: sort` runs ByteTrack's own SORT re-implementation (`yolox/sort_tracker/sort.py`), the
"SORT" row of the ByteTrack paper's Table 2 (74.6 MOTA / 76.9 IDF1), with `tools/track_sort.py` defaults:
det_thresh 0.4, max_age 30, min_hits 3, IoU 0.3, updated on every frame.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

import src.bt_compat  # noqa: F401
from yolox.tracker.byte_tracker import BYTETracker, STrack
from yolox.tracker.kalman_filter import KalmanFilter

DEFAULT_TRACKER = {
    "type": "byte",
    "track_thresh": 0.6,
    "match_thresh": 0.9,  # tools/track.py default; the paper implies 0.8
    "track_buffer": 30,
    "per_seq_buffer": {"MOT17-05-FRCNN": 14, "MOT17-13-FRCNN": 25},  # hard-coded in ByteTrack's evaluator
    "min_box_area": 100,
    "aspect_ratio_thresh": 1.6,
    "mot20": False,
    "kalman_noise_scale": 1.0,
}

_KF_POS, _KF_VEL = KalmanFilter()._std_weight_position, KalmanFilter()._std_weight_velocity


def _set_kalman_scale(tracker: BYTETracker, scale: float):
    for kf in (tracker.kalman_filter, STrack.shared_kalman):
        kf._std_weight_position = _KF_POS * scale
        kf._std_weight_velocity = _KF_VEL * scale


def track_sequence(seq: str, dets: dict, tcfg: dict, logger=None) -> list[tuple]:
    """dets: output of src.detect_io.load_dets. Returns rows (frame, id, x1, y1, w, h, score).
    `logger`: optional src.track_log.AssociationLogger (read-only; call inside `logger.patch()`)."""
    cfg = {**DEFAULT_TRACKER, **(tcfg or {})}
    if cfg["type"] == "sort":
        return _track_sequence_sort(dets, cfg)
    if cfg["type"] != "byte":
        raise ValueError(f"unknown tracker type {cfg['type']!r}")
    buffer = (cfg.get("per_seq_buffer") or {}).get(seq, cfg["track_buffer"])
    args = SimpleNamespace(track_thresh=cfg["track_thresh"], track_buffer=buffer,
                           match_thresh=cfg["match_thresh"], mot20=cfg["mot20"])
    tracker = BYTETracker(args, frame_rate=30)
    _set_kalman_scale(tracker, cfg["kalman_noise_scale"])
    h, w = dets["hw"]
    rows = []
    try:
        for frame in dets["frames"]:
            d = dets["by_frame"][frame]
            if d is None:  # ByteTrack skips the update when the detector returns None
                continue
            if logger is not None:
                logger.begin(frame, tracker)
            # boxes are already in original pixels -> pass img_size = (h, w) so the tracker's scale is 1.0
            online = tracker.update(d.copy(), [h, w], (h, w))
            for t in online:
                tlwh = t.tlwh
                vertical = tlwh[2] / tlwh[3] > cfg["aspect_ratio_thresh"]
                if tlwh[2] * tlwh[3] > cfg["min_box_area"] and not vertical:
                    rows.append((frame, t.track_id, *tlwh, t.score))
    finally:
        _set_kalman_scale(tracker, 1.0)  # restore the shared class-level filter
    return rows


class _AsTensor:
    """Minimal stand-in for the CUDA tensor that Sort.update expects (it calls .cpu().numpy())."""

    def __init__(self, arr):
        self._arr = arr

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


def _track_sequence_sort(dets: dict, cfg: dict) -> list[tuple]:
    from yolox.sort_tracker.sort import KalmanBoxTracker, Sort

    KalmanBoxTracker.count = 0  # ids restart per sequence (does not change any metric)
    tracker = Sort(cfg.get("sort_det_thresh", 0.4))
    h, w = dets["hw"]
    rows = []
    for frame in dets["frames"]:
        d = dets["by_frame"][frame]
        d = np.zeros((0, 5), np.float32) if d is None else d
        # Sort expects YOLOX's 7-column output and multiplies obj * cls -> pass cls = 1 so score is unchanged
        arr = np.concatenate([d, np.ones((len(d), 1), np.float32)], axis=1)
        for t in tracker.update(_AsTensor(arr), [h, w], (h, w)):
            tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
            vertical = tlwh[2] / tlwh[3] > cfg["aspect_ratio_thresh"]
            if tlwh[2] * tlwh[3] > cfg["min_box_area"] and not vertical:
                rows.append((frame, int(t[4]), *tlwh, -1))
    return rows


def write_mot(path: Path, rows: list[tuple]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for frame, tid, x1, y1, w, h, s in rows:
            f.write(f"{frame},{tid},{round(x1, 1)},{round(y1, 1)},{round(w, 1)},{round(h, 1)},{round(float(s), 2)},-1,-1,-1\n")
