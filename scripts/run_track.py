"""Run BYTETracker on cached detections (CPU, fast) and write MOT-format results.

    uv run python scripts/run_track.py --config configs/p1_smoke.yaml

Output: results/<run_id>/tracks/<seq>.txt, config.yaml, env.json
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from src.config import det_cache_dir, load_config, run_dir  # noqa: E402
from src.detect_io import load_dets  # noqa: E402
from src.postprocess import interpolate  # noqa: E402
from src.runlog import environment, write_json  # noqa: E402
from src.track import DEFAULT_TRACKER, track_sequence, write_mot  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dets-dir", default=None, help="override detection cache folder")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = run_dir(cfg)
    dets_dir = Path(args.dets_dir) if args.dets_dir else det_cache_dir(cfg)
    (out / "tracks").mkdir(parents=True, exist_ok=True)
    resolved = {k: v for k, v in cfg.items() if not k.startswith("_")}
    resolved["source_config"] = Path(cfg["_config_path"]).as_posix()
    with open(out / "config.yaml", "w", encoding="utf-8") as f:  # fully resolved (after `base:` merge)
        yaml.safe_dump(resolved, f, sort_keys=False)

    tcfg = {**DEFAULT_TRACKER, **(cfg.get("tracker") or {})}
    interp = (cfg.get("postprocess") or {}).get("interpolation")  # e.g. {n_min: 25, n_dti: 20}
    stats = {}
    for seq in cfg["seqs"]:
        dets = load_dets(dets_dir / f"{seq}.npz")
        t0 = time.perf_counter()
        rows = track_sequence(seq, dets, tcfg)
        dt = time.perf_counter() - t0
        if interp:
            rows = interpolate(rows, n_min=interp.get("n_min", 25), n_dti=interp.get("n_dti", 20))
        write_mot(out / "tracks" / f"{seq}.txt", rows)
        stats[seq] = {"frames": len(dets["frames"]), "boxes": len(rows), "track_ms_per_frame": 1000 * dt / max(1, len(dets["frames"]))}
        print(f"{seq}: {len(rows)} boxes, {stats[seq]['track_ms_per_frame']:.2f} ms/frame (tracker)")

    write_json(out / "env.json", environment({"tracker": tcfg, "postprocess": cfg.get("postprocess"), "dets_dir": str(dets_dir), "track_stats": stats}))
    print(f"Tracks written to {out / 'tracks'}")


if __name__ == "__main__":
    main()
