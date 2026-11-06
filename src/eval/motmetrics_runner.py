"""MOTA / IDF1 exactly as ByteTrack's `tools/track.py` computes them (py-motmetrics).

ByteTrack loads gt with fmt='mot15-2D', min_confidence=1 (no class filtering beyond the
'consider' flag) and matches with IoU distance threshold 0.5. This is the protocol behind the
published 76.6 MOTA / 79.3 IDF1, so we report it next to TrackEval's numbers.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import motmetrics as mm

from src.config import VAL_HALF_DIR

METRICS = ["mota", "idf1", "num_switches", "num_false_positives", "num_misses", "motp", "mostly_tracked",
           "mostly_lost", "num_fragmentations", "num_objects", "num_predictions"]


def evaluate(run_dir: Path, seqs: list[str], gt_dir: Path = VAL_HALF_DIR, tracker_sub_folder: str = "tracks") -> dict:
    mm.lap.default_solver = "lap"
    run_dir = Path(run_dir)
    accs, names = [], []
    for seq in seqs:
        gt = mm.io.loadtxt(str(gt_dir / seq / "gt" / "gt.txt"), fmt="mot15-2D", min_confidence=1)
        ts = mm.io.loadtxt(str(run_dir / tracker_sub_folder / f"{seq}.txt"), fmt="mot15-2D", min_confidence=-1)
        accs.append(mm.utils.compare_to_groundtruth(gt, ts, "iou", distth=0.5))
        names.append(seq)
    mh = mm.metrics.create()
    summary = mh.compute_many(accs, names=names, metrics=METRICS, generate_overall=True)
    out = OrderedDict()
    for name, row in summary.iterrows():
        d = {k: float(row[k]) for k in METRICS}
        for k in ("mota", "idf1", "motp"):
            d[k] = 100 * d[k] if k != "motp" else d[k]
        out["combined" if name == "OVERALL" else name] = d
    return dict(out)
