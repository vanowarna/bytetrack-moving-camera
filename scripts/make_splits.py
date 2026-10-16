"""Task 1.1 + 1.2: build the MOT17 half-val split and check the TrackEval pipeline.

    uv run python scripts/make_splits.py            # build data/mot/val_half + self-test
    uv run python scripts/make_splits.py --no-selftest

Self-test: the val-half ground truth is written out as if it were a tracker result and scored
with TrackEval. A correct layout gives HOTA = MOTA = IDF1 = 100.
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from tabulate import tabulate  # noqa: E402

from src.config import CACHE_DIR, SEQS, VAL_HALF_DIR  # noqa: E402
from src.data.half_split import make_val_half  # noqa: E402


def selftest(seqs):
    from src.eval.trackeval_runner import evaluate

    run = CACHE_DIR / "selftest" / "gt_as_tracker"
    if run.exists():
        shutil.rmtree(run)
    (run / "tracks").mkdir(parents=True)
    for seq in seqs:
        gt = np.loadtxt(VAL_HALF_DIR / seq / "gt" / "gt.txt", delimiter=",")
        keep = (gt[:, 6] == 1) & (gt[:, 7] == 1)  # considered pedestrians
        with open(run / "tracks" / f"{seq}.txt", "w") as f:
            for r in gt[keep]:
                f.write(f"{int(r[0])},{int(r[1])},{r[2]},{r[3]},{r[4]},{r[5]},1,-1,-1,-1\n")
    res = evaluate(run, seqs, name="selftest")["combined"]
    ok = all(abs(res[k] - 100) < 1e-6 for k in ("HOTA", "MOTA", "IDF1"))
    print(f"\nTrackEval self-test (gt as tracker): HOTA={res['HOTA']:.2f} MOTA={res['MOTA']:.2f} "
          f"IDF1={res['IDF1']:.2f} -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-selftest", action="store_true")
    ap.add_argument("--no-image-check", action="store_true", help="skip counting images (faster)")
    args = ap.parse_args()

    summary = make_val_half(SEQS, check_images=not args.no_image_check)
    rows = [[s, v["n_frames"], v["val_frames"], v["first_val_image"], v["gt_rows"], v["fps"],
             f"{v['width']}x{v['height']}"] for s, v in summary.items()]
    print(tabulate(rows, headers=["seq", "frames", "val frames", "first val img", "gt rows", "fps", "res"]))
    total = sum(v["val_frames"] for v in summary.values())
    print(f"Total val frames: {total}  ->  {VAL_HALF_DIR}")
    for s, v in summary.items():  # ByteTrack convention: val = N - N//2 - 1 frames
        assert v["val_frames"] == v["n_frames"] - v["n_frames"] // 2 - 1, s
    if not args.no_selftest and not selftest(SEQS):
        sys.exit(1)


if __name__ == "__main__":
    main()
