# Where ByteTrack Breaks: the Moving-Camera Penalty on MOT17

Reproduction and analysis of [ByteTrack](https://github.com/ifzhang/ByteTrack) on the MOT17 half-validation split,
with results separated by camera type (static: MOT17-02/04/09; moving: MOT17-05/10/11/13).

Term project, 2102515 Digital Video Processing, Chulalongkorn University (Semester 1/2026).
Author: Vanodhya Warnasooriya.

> Status: work in progress.

## 1. Environment
Requires [uv](https://docs.astral.sh/uv/), git, and an NVIDIA GPU with a recent driver (tested target: RTX 50-series, CUDA 12.8 wheels).

```bash
uv sync                                   # creates .venv from pyproject.toml / uv.lock
# Windows
powershell -ExecutionPolicy Bypass -File .\scripts\setup_external.ps1
# Linux / macOS
bash scripts/setup_external.sh
```

## 2. Weights
Download `bytetrack_ablation.pth.tar` from the ByteTrack model zoo and place it at `weights/bytetrack_ablation.pth.tar`.

## 3. Data
Download [MOT17](https://motchallenge.net/data/MOT17/) and extract so that the layout is:
```
data/mot/train/MOT17-02-FRCNN/{img1,gt,det,seqinfo.ini}
data/mot/test/...
```

## 4. Check
```bash
uv run python scripts/check_env.py
```

## 5. Reproduce
```bash
uv run python scripts/make_splits.py                                  # half-val split + TrackEval self-test
uv run python scripts/run_pipeline.py --config configs/p1_smoke.yaml  # detect -> track -> evaluate (MOT17-02)
```
Steps can also be run one by one: `run_detect.py` (GPU, cached in `cache/dets/`), `run_track.py` (CPU), `run_eval.py`.
Each run writes `results/<run_id>/` with `config.yaml`, `env.json`, `tracks/`, `metrics.json`, `metrics_per_seq.csv` and TrackEval outputs.

## Credits
- ByteTrack (MIT) — Zhang et al., ECCV 2022. Used unmodified from `external/ByteTrack`; see `src/bt_compat.py` for the compatibility shims.
- TrackEval (MIT) — Luiten et al.
- MOT17 — Milan et al., MOTChallenge (data not redistributed).

## License
MIT (this repository's own code). Third-party code keeps its own license.
