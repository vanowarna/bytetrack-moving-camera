"""Live demonstration (task 7.1): the whole pipeline on ONE sequence, ending with a number from the paper.

    uv run python scripts/demo.py                         # MOT17-13 (moving camera), detector runs live on the GPU
    uv run python scripts/demo.py --cached                # reuse cached detections (no GPU, fastest)
    uv run python scripts/demo.py --seq MOT17-09-FRCNN    # a static-camera sequence

Steps: detect (YOLOX-X, FP16) -> track (BYTE) -> evaluate (TrackEval) -> compare with the paper's number
-> render a side-by-side video: input | tracks with IDs (solid) and Kalman predictions (dashed orange).
Output: demo_out/<seq>/demo.mp4 (git-ignored). Total time on an RTX 5050 Laptop: about one minute.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import GROUPS, RESULTS_DIR, ROOT, det_cache_dir, load_config  # noqa: E402
from src.data.half_split import val_frames  # noqa: E402
from src.detect_io import load_dets, save_dets  # noqa: E402

ORANGE = (52, 104, 235)  # BGR


def color(tid):
    rng = np.random.default_rng(int(tid) * 7919)
    return tuple(int(c) for c in rng.integers(60, 255, 3))


def banner(text):
    print(f"\n=== {text} ===", flush=True)


def dashed_rect(img, p1, p2, col, dash=10):
    x1, y1 = p1
    x2, y2 = p2
    for (a, b) in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
        n = max(1, int(np.hypot(b[0] - a[0], b[1] - a[1]) // dash))
        for i in range(0, n, 2):
            s = (int(a[0] + (b[0] - a[0]) * i / n), int(a[1] + (b[1] - a[1]) * i / n))
            e = (int(a[0] + (b[0] - a[0]) * min(i + 1, n) / n), int(a[1] + (b[1] - a[1]) * min(i + 1, n) / n))
            cv2.line(img, s, e, col, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="MOT17-13-FRCNN")
    ap.add_argument("--config", default="configs/e1_default.yaml")
    ap.add_argument("--cached", action="store_true", help="use cached detections instead of running the detector")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--no-open", action="store_true", help="do not open the video when done")
    ap.add_argument("--video-frames", type=int, default=None, help="render only the first N frames (testing)")
    args = ap.parse_args()
    t_start = time.perf_counter()
    cfg = load_config(ROOT / args.config)
    seq = args.seq
    group = next(g for g, m in GROUPS.items() if seq in m)
    out = ROOT / "demo_out" / seq
    run = out / "demo_run"
    (run / "tracks").mkdir(parents=True, exist_ok=True)
    frames = list(val_frames(seq))
    print(f"Sequence {seq} ({group} camera), {len(frames)} val-half frames")

    # 1. detection ----------------------------------------------------------------------------------
    if args.cached:
        banner("1/4 Detection: using cached YOLOX-X detections")
        dets = load_dets(det_cache_dir(cfg) / f"{seq}.npz")
    else:
        banner("1/4 Detection: YOLOX-X, FP16, live on the GPU")
        import src.torch_setup  # noqa: F401  (Windows cuDNN DLL fix, side effect only)
        from tqdm import tqdm
        from src.detect import Detector
        d = cfg["detector"]
        det = Detector(ROOT / d["exp"], ROOT / d["ckpt"], test_size=d["test_size"], conf=d["conf"], nms=d["nms"],
                       fp16=d["fp16"], fuse=d["fuse"], cudnn=d.get("cudnn", True))
        fids, dl, ms, hw = [], [], [], None
        for fid, path in tqdm(frames, unit="frame"):
            img = cv2.imread(str(path))
            hw = img.shape[:2]
            r, t = det(img)
            fids.append(fid)
            dl.append(r)
            ms.append(t)
        save_dets(out / "dets.npz", fids, dl, ms, hw)
        dets = load_dets(out / "dets.npz")
        print(f"detector: {np.mean(ms[10:]):.1f} ms/frame")

    # 2. tracking (with the read-only Kalman log, used only for drawing) -----------------------------
    banner("2/4 Tracking: BYTE (official settings)")
    from src.track import DEFAULT_TRACKER, track_sequence, write_mot
    from src.track_log import AssociationLogger
    logger = AssociationLogger()
    with logger.patch():
        rows = track_sequence(seq, dets, {**DEFAULT_TRACKER, **(cfg.get("tracker") or {})}, logger=logger)
    write_mot(run / "tracks" / f"{seq}.txt", rows)
    print(f"{len(rows)} track boxes, {len({r[1] for r in rows})} identities")

    # 3. evaluation -----------------------------------------------------------------------------------
    banner("3/4 Evaluation: TrackEval on the val half")
    from src.eval.trackeval_runner import evaluate
    res = evaluate(run, [seq], name="demo")["per_seq"][seq]
    paper = json.loads((RESULTS_DIR / "e1_default" / "metrics.json").read_text(encoding="utf-8"))["trackeval_per_seq"][seq]
    print(f"{'':10s}{'demo':>8s}{'paper':>8s}")
    for k in ("HOTA", "DetA", "AssA", "MOTA", "IDF1"):
        print(f"{k:10s}{res[k]:8.1f}{paper[k]:8.1f}   {'match' if abs(res[k] - paper[k]) < 0.05 else 'DIFFERENT'}")
    print(f"{'IDSW':10s}{res['IDSW']:8d}{paper['IDSW']:8d}")

    # 4. video ------------------------------------------------------------------------------------------
    if not args.no_video:
        banner("4/4 Video: input | tracks + Kalman predictions")
        tr = pd.DataFrame(rows, columns=["frame", "id", "x", "y", "w", "h", "s"])
        log = logger.to_frame()
        pred = log[(log.stage == "high") & (log.state == "tracked")]
        cam_p = RESULTS_DIR / "e3" / "camera_motion" / f"{seq}.csv"
        cam = pd.read_csv(cam_p).set_index("frame")["disp_pct"] if cam_p.exists() else None
        first = cv2.imread(str(frames[0][1]))
        H, W = first.shape[:2]
        sc = 960 / W
        size = (int(W * sc) * 2, int(H * sc))
        fps = 25 if "13" in seq else (14 if "05" in seq else 30)
        vw = cv2.VideoWriter(str(out / "demo.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        for fid, path in frames[: args.video_frames]:
            img = cv2.imread(str(path))
            right = img.copy()
            for _, t in tr[tr.frame == fid].iterrows():
                c = color(t.id)
                cv2.rectangle(right, (int(t.x), int(t.y)), (int(t.x + t.w), int(t.y + t.h)), c, 3)
                cv2.putText(right, str(int(t.id)), (int(t.x), int(t.y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, c, 2)
            for _, p in pred[pred.frame == fid].iterrows():
                dashed_rect(right, (int(p.px1), int(p.py1)), (int(p.px2), int(p.py2)), ORANGE)
            left, right = cv2.resize(img, (size[0] // 2, size[1])), cv2.resize(right, (size[0] // 2, size[1]))
            info = f"{seq}  frame {fid}"
            if cam is not None and fid in cam.index and np.isfinite(cam[fid]):
                info += f"  camera motion {cam[fid]:.2f}% diag/frame"
            for im, txt in ((left, "input"), (right, "ByteTrack: tracks (solid), Kalman predictions (dashed)")):
                cv2.rectangle(im, (0, 0), (im.shape[1], 30), (0, 0, 0), -1)
                cv2.putText(im, txt, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(left, info, (8, size[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            vw.write(np.hstack([left, right]))
        vw.release()
        print(f"video: {out / 'demo.mp4'}")
        if not args.no_open and sys.platform == "win32":
            import os
            os.startfile(str(out / "demo.mp4"))  # noqa: S606
    print(f"\nDone in {time.perf_counter() - t_start:.0f} s")


if __name__ == "__main__":
    main()
