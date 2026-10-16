"""Run the official BYTETracker on cached detections, reproducing ByteTrack's evaluator.

Faithful to `yolox/evaluators/mot_evaluator.py` (MOTEvaluator.evaluate):
- a new BYTETracker(args, frame_rate=30) per sequence
- per-sequence track_buffer override (MOT17-05 -> 14, MOT17-13 -> 25, else 30) when enabled
- tracker.update is NOT called on frames where the detector returned None
- output filter: keep a track box if w*h > min_box_area and w/h <= aspect_ratio_thresh
- result line: frame,id,x1,y1,w,h,score,-1,-1,-1 (box rounded to 0.1, score to 0.01)
Only addition: an optional Kalman noise scale for the sensitivity study (default 1.0 = official).
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


import src.bt_compat  # noqa: F401
from yolox.tracker.byte_tracker import BYTETracker, STrack
from yolox.tracker.kalman_filter import KalmanFilter

DEFAULT_TRACKER = {
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


def track_sequence(seq: str, dets: dict, tcfg: dict) -> list[tuple]:
    """dets: output of src.detect.load_dets. Returns rows (frame, id, x1, y1, w, h, score)."""
    cfg = {**DEFAULT_TRACKER, **(tcfg or {})}
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


def write_mot(path: Path, rows: list[tuple]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for frame, tid, x1, y1, w, h, s in rows:
            f.write(f"{frame},{tid},{round(x1, 1)},{round(y1, 1)},{round(w, 1)},{round(h, 1)},{round(float(s), 2)},-1,-1,-1\n")
