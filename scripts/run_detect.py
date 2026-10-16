"""Run the YOLOX detector on the val half and cache detections (GPU step, done once per detector setting).

    uv run python scripts/run_detect.py --config configs/p1_smoke.yaml
    uv run python scripts/run_detect.py --config configs/p1_smoke.yaml --max-frames 20   # quick test

Output: cache/dets/<det_id>/<seq>.npz + meta.json (timing, versions, weights hash).
Existing files are skipped unless --force.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.config import ROOT, det_cache_dir, load_config  # noqa: E402
from src.data.half_split import val_frames  # noqa: E402
from src.runlog import environment, sha256, write_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-frames", type=int, default=None, help="only the first N val frames per sequence (testing)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dcfg = cfg["detector"]
    out_dir = det_cache_dir(cfg)
    if args.max_frames:
        out_dir = out_dir.with_name(out_dir.name + f"_first{args.max_frames}")
    todo = [s for s in cfg["seqs"] if args.force or not (out_dir / f"{s}.npz").exists()]
    if not todo:
        print(f"All detections already cached in {out_dir}")
        return

    import src.torch_setup  # noqa: F401  (Windows cuDNN DLL fix)
    import torch
    from src.detect import Detector
    from src.detect_io import save_dets

    torch.manual_seed(cfg.get("seed", 0))
    det = Detector(ROOT / dcfg["exp"], ROOT / dcfg["ckpt"], test_size=dcfg["test_size"], conf=dcfg["conf"],
                   nms=dcfg["nms"], fp16=dcfg["fp16"], fuse=dcfg["fuse"], cudnn=dcfg.get("cudnn", True))
    timing = {}
    for seq in todo:
        frames, dets, ms, hw = [], [], [], None
        items = list(val_frames(seq))[: args.max_frames]
        for fid, path in tqdm(items, desc=seq, unit="frame"):
            img = cv2.imread(str(path))
            if img is None:
                raise FileNotFoundError(path)
            hw = img.shape[:2]
            d, t = det(img)
            frames.append(fid)
            dets.append(d)
            ms.append(t)
        save_dets(out_dir / f"{seq}.npz", frames, dets, ms, hw)
        warm = np.asarray(ms[10:] or ms)
        timing[seq] = {"frames": len(frames), "ms_mean": float(warm.mean()), "ms_median": float(np.median(warm))}
        print(f"{seq}: {len(frames)} frames, {timing[seq]['ms_mean']:.1f} ms/frame (detector, excl. 10 warm-up)")

    meta_path = out_dir / "meta.json"
    write_json(meta_path, environment({"detector": dcfg, "weights_sha256": sha256(ROOT / dcfg["ckpt"]),
                                       "timing": timing, "max_frames": args.max_frames}))
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
