"""E3 task 5.3: re-run the tracker with the read-only association logger, and extract per-frame
error events (ID switches, misses) with ByteTrack's own evaluation protocol (py-motmetrics).

    uv run python scripts/run_e3_log.py --config configs/e1_default.yaml

Checks that the logged run produces exactly the tracks in results/<run_id>/tracks (logger is read-only).
Output: results/e3/assoc/<seq>.csv.gz  (one row per track per association stage per frame)
        results/e3/events/<seq>.csv    (per frame: gt, match, switch, miss, fp)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import motmetrics as mm  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import RESULTS_DIR, VAL_HALF_DIR, det_cache_dir, load_config, run_dir  # noqa: E402
from src.detect_io import load_dets  # noqa: E402
from src.track import DEFAULT_TRACKER, track_sequence, write_mot  # noqa: E402
from src.track_log import AssociationLogger  # noqa: E402


def canonical(text: str) -> str:
    """Relabel track ids by order of first appearance. ByteTrack's id counter is global (it keeps counting
    across sequences), so absolute ids depend on which sequences ran before; boxes and id structure do not."""
    ids, out = {}, []
    for line in text.splitlines():
        f = line.split(",")
        f[1] = str(ids.setdefault(f[1], len(ids) + 1))
        out.append(",".join(f))
    return "\n".join(out)


def frame_events(gt_file: Path, ts_file: Path) -> pd.DataFrame:
    mm.lap.default_solver = "lap"
    gt = mm.io.loadtxt(str(gt_file), fmt="mot15-2D", min_confidence=1)
    ts = mm.io.loadtxt(str(ts_file), fmt="mot15-2D", min_confidence=-1)
    acc = mm.utils.compare_to_groundtruth(gt, ts, "iou", distth=0.5)
    ev = acc.mot_events.reset_index()
    ev = ev[ev["Type"].isin(["MATCH", "SWITCH", "MISS", "FP"])]
    c = ev.groupby(["FrameId", "Type"]).size().unstack(fill_value=0)
    for k in ("MATCH", "SWITCH", "MISS", "FP"):
        if k not in c:
            c[k] = 0
    c = c.rename(columns=str.lower)[["match", "switch", "miss", "fp"]]
    c["gt"] = c["match"] + c["switch"] + c["miss"]
    return c.rename_axis("frame").reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    tcfg = {**DEFAULT_TRACKER, **(cfg.get("tracker") or {})}
    ref_dir = run_dir(cfg) / "tracks"
    out = RESULTS_DIR / "e3"
    (out / "assoc").mkdir(parents=True, exist_ok=True)
    (out / "events").mkdir(parents=True, exist_ok=True)
    tmp = out / "_check_tracks"

    for seq in cfg["seqs"]:
        dets = load_dets(det_cache_dir(cfg) / f"{seq}.npz")
        logger = AssociationLogger()
        with logger.patch():
            rows = track_sequence(seq, dets, tcfg, logger=logger)
        write_mot(tmp / f"{seq}.txt", rows)
        same = canonical((tmp / f"{seq}.txt").read_text()) == canonical((ref_dir / f"{seq}.txt").read_text())
        if not same:
            raise SystemExit(f"{seq}: logged run differs from {ref_dir} — logger must be read-only")
        df = logger.to_frame()
        df.to_csv(out / "assoc" / f"{seq}.csv.gz", index=False, float_format="%.6g")
        ev = frame_events(VAL_HALF_DIR / seq / "gt" / "gt.txt", ref_dir / f"{seq}.txt")
        ev.to_csv(out / "events" / f"{seq}.csv", index=False)
        hi = df[(df.stage == "high") & (df.state == "tracked") & (df.since == 1) & (df.matched == 1)]
        print(f"{seq}: tracks identical OK | {len(df)} log rows | one-step pred IoU median {hi.iou.median():.3f} | "
              f"switches {int(ev.switch.sum())}")
    for f in tmp.glob("*.txt"):
        f.unlink()
    tmp.rmdir()
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
