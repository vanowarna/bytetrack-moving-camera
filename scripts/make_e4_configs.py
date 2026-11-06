"""E4 (sensitivity study): write one config per setting, changing ONE tracker parameter at a time
around the official ByteTrack settings (configs/e1_default.yaml).

    uv run python scripts/make_e4_configs.py
    uv run python scripts/run_many.py --configs "configs/e4/*.yaml" --table e4     # CPU only, cached detections
    uv run python scripts/analyze_e4.py

Purpose (instructor's note): measure whether the static/moving gap is stable across settings.
It is NOT an attempt to close the gap.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from src.config import ROOT  # noqa: E402

DEFAULTS = {"track_thresh": 0.6, "match_thresh": 0.9, "track_buffer": 30, "kalman_noise_scale": 1.0}
GRID = {
    "track_thresh": [0.4, 0.5, 0.7],          # high-score detection threshold (BYTE's tau)
    "match_thresh": [0.7, 0.8, 0.95],         # first-association gate (cost = 1 - IoU * score)
    "track_buffer": [10, 15, 60, 90],         # frames a lost track is kept (30 fps reference)
    "kalman_noise_scale": [0.25, 0.5, 2.0, 4.0],  # std scale of the Kalman PROCESS noise (measurement noise fixed)
}
LABEL = {"track_thresh": "high-score thr.", "match_thresh": "match thr.", "track_buffer": "track buffer",
         "kalman_noise_scale": "Kalman process noise x"}


def main():
    out = ROOT / "configs" / "e4"
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for param, values in GRID.items():
        for v in values:
            tracker = {param: v}
            if param == "track_buffer":  # keep ByteTrack's per-sequence rule: buffer scales with frame rate
                tracker["per_seq_buffer"] = {"MOT17-05-FRCNN": round(v * 14 / 30), "MOT17-13-FRCNN": round(v * 25 / 30)}
            tag = str(v).replace(".", "p")
            cfg = {"base": "../e1_default.yaml", "run_id": f"e4_{param}_{tag}", "note": f"{LABEL[param]} = {v}",
                   "e4": {"param": param, "value": v, "default": DEFAULTS[param]}, "tracker": tracker}
            with open(out / f"e4_{param}_{tag}.yaml", "w", encoding="utf-8") as f:
                f.write("# E4 sensitivity: one parameter changed from the official settings (generated)\n")
                yaml.safe_dump(cfg, f, sort_keys=False)
            n += 1
    print(f"{n} configs written to {out}")


if __name__ == "__main__":
    main()
