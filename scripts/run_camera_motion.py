"""E3 task 5.1: estimate camera motion for every val-half frame (CPU, ~2 min for all 7 sequences).

    uv run python scripts/run_camera_motion.py --config configs/e1_default.yaml

Uses the cached detections of the config's detector to mask people out of the feature matching.
Output: results/e3/camera_motion/<seq>.csv (one row per frame; frame 1 has no previous frame).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.config import RESULTS_DIR, TRAIN_DIR, det_cache_dir, load_config  # noqa: E402
from src.data.half_split import read_seqinfo, val_frames  # noqa: E402
from src.detect_io import load_dets  # noqa: E402
from src.motion.camera_motion import CameraMotionEstimator, describe  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--work-width", type=int, default=960)
    ap.add_argument("--max-frames", type=int, default=None, help="testing only")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_dir = RESULTS_DIR / "e3" / "camera_motion"
    out_dir.mkdir(parents=True, exist_ok=True)

    for seq in cfg["seqs"]:
        info = read_seqinfo(TRAIN_DIR / seq)
        w, h, fps = int(info["imWidth"]), int(info["imHeight"]), float(info["frameRate"])
        dets = load_dets(det_cache_dir(cfg) / f"{seq}.npz")["by_frame"]
        est = CameraMotionEstimator(work_width=args.work_width)
        rows, A_prev = [], None
        for fid, path in tqdm(list(val_frames(seq))[: args.max_frames], desc=seq, unit="frame"):
            r = est.step(cv2.imread(str(path)), dets.get(fid))
            A = r.pop("A")
            row = {"frame": fid, **r, **describe(A, A_prev, w, h, fps)}
            if A is not None:
                row.update({f"a{i}{j}": A[i, j] for i in range(2) for j in range(3)})
            rows.append(row)
            A_prev = A
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / f"{seq}.csv", index=False, float_format="%.6g")
        ok = df["ok"].iloc[1:].mean() * 100
        print(f"{seq}: fit ok {ok:.0f}% | median motion {df['disp_pct'].median():.3f} %diag/frame | "
              f"p90 {df['disp_pct'].quantile(0.9):.3f} | median accel {df['accel_pct'].median():.3f}")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
