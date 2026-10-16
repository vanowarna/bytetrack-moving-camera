"""Compatibility layer for running the official ByteTrack code on a modern stack
(Python 3.11, NumPy 1.26, PyTorch 2.7, Windows/Linux) without compiling C extensions.

Import this module BEFORE importing anything from `yolox`:

    import src.bt_compat  # noqa: F401
    from yolox.tracker.byte_tracker import BYTETracker

What it does (and nothing else):
1. Adds external/ByteTrack to sys.path (the repo is not pip-installed).
2. Restores removed NumPy aliases (np.float, np.int, np.bool) used by ByteTrack.
3. Provides `cython_bbox.bbox_overlaps` in pure NumPy (same IoU formula, +1 pixel convention).
4. Stubs `yolox._C` (only needed by a fast COCO evaluator we do not use).
5. Points YOLOX_DATADIR to ./data so ByteTrack finds data/mot/...
"""
import os
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BYTETRACK_DIR = ROOT / "external" / "ByteTrack"

if not BYTETRACK_DIR.exists():
    raise FileNotFoundError(
        f"{BYTETRACK_DIR} not found. Run scripts/setup_external.ps1 (Windows) or scripts/setup_external.sh first."
    )
if str(BYTETRACK_DIR) not in sys.path:
    sys.path.insert(0, str(BYTETRACK_DIR))

# 2. NumPy aliases removed in 1.24
for _name, _typ in (("float", float), ("int", int), ("bool", bool)):
    if not hasattr(np, _name):
        setattr(np, _name, _typ)


# 3. cython_bbox replacement
def _bbox_overlaps(boxes: np.ndarray, query_boxes: np.ndarray) -> np.ndarray:
    """IoU between (N,4) and (K,4) boxes in tlbr format, matching cython_bbox (+1 convention)."""
    boxes = np.asarray(boxes, dtype=np.float64)
    query_boxes = np.asarray(query_boxes, dtype=np.float64)
    if boxes.size == 0 or query_boxes.size == 0:
        return np.zeros((len(boxes), len(query_boxes)), dtype=np.float64)
    b = boxes[:, None, :]
    q = query_boxes[None, :, :]
    iw = np.minimum(b[..., 2], q[..., 2]) - np.maximum(b[..., 0], q[..., 0]) + 1
    ih = np.minimum(b[..., 3], q[..., 3]) - np.maximum(b[..., 1], q[..., 1]) + 1
    inter = np.clip(iw, 0, None) * np.clip(ih, 0, None)
    area_b = (b[..., 2] - b[..., 0] + 1) * (b[..., 3] - b[..., 1] + 1)
    area_q = (q[..., 2] - q[..., 0] + 1) * (q[..., 3] - q[..., 1] + 1)
    union = area_b + area_q - inter
    return np.where(inter > 0, inter / union, 0.0)


if "cython_bbox" not in sys.modules:
    _cb = types.ModuleType("cython_bbox")
    _cb.bbox_overlaps = _bbox_overlaps
    sys.modules["cython_bbox"] = _cb

# 4. yolox._C stub
sys.modules.setdefault("yolox._C", types.ModuleType("yolox._C"))

# 5. dataset root
os.environ.setdefault("YOLOX_DATADIR", str(ROOT / "data"))
