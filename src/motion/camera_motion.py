"""Global (camera) motion between consecutive frames: ORB features + RANSAC similarity fit (Lecture 4).

For each pair (t-1 -> t) we fit a 4-DOF similarity transform A_t (rotation, uniform scale, translation)
with cv2.estimateAffinePartial2D. People are masked out using the detector's own boxes, so the fit
follows the background. Frames are downscaled to at most `work_width` pixels for speed; the
returned transform is in original pixel coordinates.

Per frame we report
    disp_px  : mean displacement of the 4 image corners under A_t (camera speed, px/frame)
    accel_px : mean displacement of the corners under A_t - A_{t-1} (change of camera motion, px/frame^2)
both also as % of the image diagonal. A constant-velocity Kalman filter in image coordinates can absorb a
steady pan (its velocity learns it) but not a change of camera motion, so `accel` is the quantity that
should hurt the prediction most.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

IDENTITY = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def _mask(shape, boxes, f, min_score=0.3, pad=0.1):
    m = np.full(shape, 255, np.uint8)
    if boxes is None or len(boxes) == 0:
        return m
    h, w = shape
    for x1, y1, x2, y2, s in boxes:
        if s < min_score:
            continue
        bw, bh = (x2 - x1) * pad, (y2 - y1) * pad
        xa, ya = max(0, int((x1 - bw) * f)), max(0, int((y1 - bh) * f))
        xb, yb = min(w, int((x2 + bw) * f) + 1), min(h, int((y2 + bh) * f) + 1)
        m[ya:yb, xa:xb] = 0
    return m


def corners(w: int, h: int) -> np.ndarray:
    return np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float64)


def apply(A: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ A[:, :2].T + A[:, 2]


class CameraMotionEstimator:
    def __init__(self, work_width: int = 960, n_features: int = 3000, ratio: float = 0.75,
                 ransac_px: float = 3.0, min_inliers: int = 15, seed: int = 0):
        self.work_width, self.ratio, self.ransac_px, self.min_inliers = work_width, ratio, ransac_px, min_inliers
        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        cv2.setRNGSeed(seed)
        self.prev = None  # (kp, desc)
        self.f = 1.0

    def reset(self):
        self.prev = None

    def _features(self, img_bgr, boxes):
        h, w = img_bgr.shape[:2]
        self.f = min(1.0, self.work_width / w)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        if self.f < 1.0:
            gray = cv2.resize(gray, (round(w * self.f), round(h * self.f)), interpolation=cv2.INTER_AREA)
        return self.orb.detectAndCompute(gray, _mask(gray.shape, boxes, self.f))

    def step(self, img_bgr: np.ndarray, boxes: np.ndarray | None) -> dict:
        """Feed the next frame (+ its detections for masking). Returns the transform from the previous frame."""
        kp, desc = self._features(img_bgr, boxes)
        prev, self.prev = self.prev, (kp, desc)
        out = {"ok": False, "n_kp": len(kp), "n_good": 0, "n_inliers": 0, "A": None}
        if prev is None or desc is None or prev[1] is None or len(kp) < 2 or len(prev[0]) < 2:
            return out
        pairs = self.matcher.knnMatch(prev[1], desc, k=2)
        good = [m for m, *n in pairs if n and m.distance < self.ratio * n[0].distance]
        out["n_good"] = len(good)
        if len(good) < self.min_inliers:
            return out
        p1 = np.float32([prev[0][m.queryIdx].pt for m in good])
        p2 = np.float32([kp[m.trainIdx].pt for m in good])
        A, inl = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC, ransacReprojThreshold=self.ransac_px,
                                             maxIters=2000, confidence=0.995)
        n_in = int(inl.sum()) if inl is not None else 0
        out["n_inliers"] = n_in
        if A is None or n_in < self.min_inliers:
            return out
        A = A.astype(np.float64)
        A[:, 2] /= self.f  # translation back to original pixels (linear part is scale-free)
        out.update(ok=True, A=A)
        return out


def describe(A: np.ndarray | None, A_prev: np.ndarray | None, w: int, h: int, fps: float) -> dict:
    """Scalar summaries of one transform (and of its change from the previous one)."""
    diag = math.hypot(w, h)
    nan = float("nan")
    if A is None:
        return {k: nan for k in ("dx", "dy", "scale", "rot_deg", "disp_px", "disp_pct", "disp_pct_s",
                                 "accel_px", "accel_pct")}
    c = corners(w, h)
    flow = apply(A, c) - c
    disp = float(np.linalg.norm(flow, axis=1).mean())
    if A_prev is not None:
        flow_prev = apply(A_prev, c) - c
        accel = float(np.linalg.norm(flow - flow_prev, axis=1).mean())
    else:
        accel = nan
    return {
        "dx": float(A[0, 2]), "dy": float(A[1, 2]),
        "scale": float(math.hypot(A[0, 0], A[1, 0])), "rot_deg": float(math.degrees(math.atan2(A[1, 0], A[0, 0]))),
        "disp_px": disp, "disp_pct": 100 * disp / diag, "disp_pct_s": 100 * disp / diag * fps,
        "accel_px": accel, "accel_pct": 100 * accel / diag if accel == accel else nan,
    }


def box_flow(A: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Image displacement that the camera motion A causes at the given points (N,2) -> (N,2)."""
    return apply(A, centers) - centers
