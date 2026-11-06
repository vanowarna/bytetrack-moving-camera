"""YOLOX detector wrapper that reproduces ByteTrack's inference path exactly.

Matches `yolox/evaluators/mot_evaluator.py` + `tools/track.py`:
- preproc: keep-ratio resize to test_size, pad 114, BGR->RGB, /255, ImageNet mean/std
- model in FP16 (optional) with fused conv+bn (optional)
- yolox.utils.postprocess(conf, nms) on the raw output (same dtype as the model)
- score = obj_conf * cls_conf, boxes divided by the resize ratio, computed in the
  output dtype (float16 when fp16) exactly like BYTETracker.update does.
The result is stored as float32 [x1, y1, x2, y2, score] in original image pixels, so
tracking from the cache gives the same numbers as tracking online.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import src.bt_compat  # noqa: F401  (must precede yolox imports)
import src.torch_setup  # noqa: F401  (Windows cuDNN DLL fix, must precede torch)
import torch
from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, postprocess

from src.detect_io import load_dets, save_dets  # noqa: F401  (re-export)

RGB_MEANS = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class Detector:
    def __init__(self, exp_file: str | Path, ckpt: str | Path, test_size=(800, 1440), conf: float = 0.01,
                 nms: float = 0.7, fp16: bool = True, fuse: bool = True, device: str = "cuda",
                 cudnn: bool = True):
        torch.backends.cudnn.enabled = cudnn  # fallback switch if cuDNN cannot load
        torch.backends.cudnn.benchmark = False
        exp = get_exp(str(exp_file), None)
        exp.test_size = tuple(test_size)
        exp.test_conf = conf
        exp.nmsthre = nms
        self.test_size = tuple(test_size)
        self.conf, self.nms, self.num_classes = conf, nms, exp.num_classes
        self.fp16, self.device = fp16, device

        model = exp.get_model()
        state = torch.load(str(ckpt), map_location="cpu", weights_only=False)  # official checkpoint (trusted)
        model.load_state_dict(state["model"])
        model.to(device).eval()
        if fuse:
            model = fuse_model(model)
        if fp16:
            model = model.half()
        self.model = model

    def _sync(self):
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()

    @torch.no_grad()
    def __call__(self, img_bgr: np.ndarray) -> tuple[np.ndarray | None, float]:
        """Return (dets, ms). dets is float32 (N,5) [x1,y1,x2,y2,score] or None if no detection."""
        x, ratio = preproc(img_bgr, self.test_size, RGB_MEANS, STD)
        t = torch.from_numpy(x).unsqueeze(0).to(self.device)
        t = t.half() if self.fp16 else t.float()
        self._sync()
        t0 = time.perf_counter()
        out = self.model(t)
        out = postprocess(out, self.num_classes, self.conf, self.nms)[0]
        self._sync()
        ms = (time.perf_counter() - t0) * 1000.0
        if out is None:
            return None, ms
        out = out.cpu().numpy()  # float16 if fp16, like ByteTrack
        scores = out[:, 4] * out[:, 5]
        h, w = img_bgr.shape[:2]
        scale = min(self.test_size[0] / float(h), self.test_size[1] / float(w))
        boxes = out[:, :4]
        boxes /= scale  # in place, same dtype as ByteTrack's tracker
        return np.concatenate([boxes, scores[:, None]], axis=1).astype(np.float32), ms
