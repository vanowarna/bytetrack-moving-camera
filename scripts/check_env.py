"""Ground check: GPU, ByteTrack import, weights and dataset layout.
Run:  uv run python scripts/check_env.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ok = True

def report(name, cond, hint=""):
    global ok
    ok &= bool(cond)
    print(f"[{'OK' if cond else 'FAIL'}] {name}" + ("" if cond else f"  -> {hint}"))

import torch
report(f"torch {torch.__version__}", True)
cuda = torch.cuda.is_available()
report("CUDA available", cuda, "install the cu128 torch wheel (uv sync) and update the NVIDIA driver")
if cuda:
    print(f"      GPU: {torch.cuda.get_device_name(0)}  capability: {torch.cuda.get_device_capability(0)}")
    x = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    report("fp16 matmul on GPU", torch.isfinite(x @ x).all().item())

try:
    import src.bt_compat  # noqa: F401
    from yolox.tracker.byte_tracker import BYTETracker  # noqa: F401
    from yolox.exp import get_exp  # noqa: F401
    report("ByteTrack import (via src/bt_compat.py)", True)
except Exception as e:  # noqa: BLE001
    report("ByteTrack import (via src/bt_compat.py)", False, repr(e))

try:
    import trackeval  # noqa: F401
    report("TrackEval import", True)
except Exception as e:  # noqa: BLE001
    report("TrackEval import", False, f"{e!r} (run scripts/setup_external.*)")

w = ROOT / "weights" / "bytetrack_ablation.pth.tar"
report("weights/bytetrack_ablation.pth.tar", w.exists(), "download from the ByteTrack README model zoo")

seqs = ["02", "04", "05", "09", "10", "11", "13"]
train = ROOT / "data" / "mot" / "train"
missing = [s for s in seqs if not (train / f"MOT17-{s}-FRCNN" / "img1").exists()]
report("data/mot/train/MOT17-XX-FRCNN (7 seqs)", not missing, f"missing: {missing}")

print("\nALL GOOD" if ok else "\nFix the FAIL lines above.")
