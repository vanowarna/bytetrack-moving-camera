"""Evaluate a run with TrackEval (HOTA, CLEAR, Identity) on the MOT17 val half.

Expected layout:
    results/<run_id>/tracks/<seq>.txt         (tracker output, MOT format)
    data/mot/val_half/<seq>/gt/gt.txt          (from scripts/make_splits.py)
TrackEval's standard MOT17 preprocessing is used (pedestrian class, distractors removed,
zero-marked gt ignored), so numbers are comparable with published TrackEval results.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import src.np_compat  # noqa: F401  (TrackEval uses np.float / np.int)
import trackeval

from src.config import VAL_HALF_DIR
from src.data.half_split import val_length

COMBINED = "COMBINED_SEQ"


def _summary(res: dict) -> dict:
    """Flatten one TrackEval result (one sequence or combined, class 'pedestrian')."""
    h, c, i = res["HOTA"], res["CLEAR"], res["Identity"]
    return {
        "HOTA": 100 * float(np.mean(h["HOTA"])),
        "DetA": 100 * float(np.mean(h["DetA"])),
        "AssA": 100 * float(np.mean(h["AssA"])),
        "LocA": 100 * float(np.mean(h["LocA"])),
        "MOTA": 100 * float(c["MOTA"]),
        "MOTP": 100 * float(c["MOTP"]),
        "IDF1": 100 * float(i["IDF1"]),
        "IDSW": int(c["IDSW"]),
        "FP": int(c["CLR_FP"]),
        "FN": int(c["CLR_FN"]),
        "TP": int(c["CLR_TP"]),
        "Frag": int(c["Frag"]),
        "MT": int(c["MT"]),
        "ML": int(c["ML"]),
    }


def evaluate(run_dir: Path, seqs: list[str], name: str = "all", gt_dir: Path = VAL_HALF_DIR,
             tracker_sub_folder: str = "tracks") -> dict:
    """Return {'per_seq': {seq: summary}, 'combined': summary}. Writes TrackEval files to run_dir/trackeval/<name>/."""
    run_dir = Path(run_dir)
    tracker = run_dir.name
    dataset_config = {
        "GT_FOLDER": str(gt_dir),
        "TRACKERS_FOLDER": str(run_dir.parent),
        "TRACKERS_TO_EVAL": [tracker],
        "TRACKER_SUB_FOLDER": tracker_sub_folder,
        "OUTPUT_FOLDER": str(run_dir.parent),
        "OUTPUT_SUB_FOLDER": f"trackeval/{name}",  # -> results/<run_id>/trackeval/<name>/
        "BENCHMARK": "MOT17",
        "SPLIT_TO_EVAL": "train",
        "SKIP_SPLIT_FOL": True,
        "SEQ_INFO": {s: val_length(s, gt_dir) for s in seqs},
        "CLASSES_TO_EVAL": ["pedestrian"],
        "DO_PREPROC": True,
        "PRINT_CONFIG": False,
    }
    eval_config = {
        "USE_PARALLEL": False,
        "PRINT_RESULTS": False,
        "PRINT_CONFIG": False,
        "TIME_PROGRESS": False,
        "DISPLAY_LESS_PROGRESS": True,
        "OUTPUT_SUMMARY": True,
        "OUTPUT_DETAILED": True,
        "PLOT_CURVES": False,
        "BREAK_ON_ERROR": True,
        "RETURN_ON_ERROR": False,
        "LOG_ON_ERROR": str(run_dir / "trackeval" / "error_log.txt"),
    }
    evaluator = trackeval.Evaluator(eval_config)
    dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
    metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR({"THRESHOLD": 0.5, "PRINT_CONFIG": False}),
               trackeval.metrics.Identity({"THRESHOLD": 0.5, "PRINT_CONFIG": False})]
    output_res, _ = evaluator.evaluate([dataset], metrics)
    res = output_res["MotChallenge2DBox"][tracker]
    out = {"per_seq": {}, "combined": None}
    for key, val in res.items():
        s = _summary(val["pedestrian"])
        if key == COMBINED:
            out["combined"] = s
        else:
            out["per_seq"][key] = s
    return out
