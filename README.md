# ByteTrack Under Camera Motion: Reproduction and Error Analysis on MOT17

**Project page:** https://vanowarna.com/bytetrack-moving-camera/ · **Paper:** [PDF](site/assets/paper.pdf) · **Slides:** [PDF](site/assets/slides.pdf)

Reproduction and analysis of [ByteTrack](https://github.com/ifzhang/ByteTrack) on the MOT17 half-validation split,
with results separated by camera type (static: MOT17-02/04/09; moving: MOT17-05/10/11/13).

Term project, 2102515 Digital Video Processing, Chulalongkorn University (Semester 1/2026).
Author: Vanodhya Warnasooriya.

| Finding | Result |
|---|---|
| E1: reproduction (MOTA / IDF1) | 76.6 / 79.3, equal to the paper (code default match threshold 0.9) |
| E2: HOTA static vs moving | 71.2 vs 60.3; gap mainly in association (AssA −13.4, DetA −8.4) |
| E3: lost tracks re-found after 9–16 frames | 2.1% moving vs 6.0% static |
| E4: gap over 15 tracker settings | 8.8–16.5 HOTA; only the Kalman process noise changes it noticeably |

## Which command makes which table or figure

| Paper item | Command (from the repository root) |
|---|---|
| Data split | `uv run python scripts/make_splits.py` |
| Table I (E1) | `uv run python scripts/run_many.py --configs "configs/e1_*.yaml" --table e1` |
| Table II (E2 columns), E2 numbers | `uv run python scripts/analyze_e2.py --run e1_default --gap-runs e1_default e1_fp32 e1_interp20 e1_match08 e1_match08_interp20 e1_sort` |
| Figs. 1–3, E3 columns of Table II | `run_camera_motion.py`, `run_e3_log.py`, `analyze_e3.py` (see E3 below) |
| Fig. 4 (E4) | `make_e4_configs.py`, `run_many.py --configs "configs/e4/*.yaml" --table e4`, `analyze_e4.py` |
| All numbers, tables and figures of the paper | `uv run python scripts/make_paper_assets.py` |
| Project page data | `uv run python scripts/make_site_data.py` |

Raw outputs of every reported run (config, environment, tracks, metrics) are in `results/`.

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
Download `bytetrack_ablation.pth.tar` from the [ByteTrack model zoo](https://github.com/ifzhang/ByteTrack#model-zoo) and place it at `weights/bytetrack_ablation.pth.tar`.

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
uv run python scripts/make_splits.py                                    # half-val split + TrackEval self-test
uv run python scripts/run_pipeline.py --config configs/e1_default.yaml  # detect -> track -> evaluate, 7 sequences
```
Steps can also be run one by one: `run_detect.py` (GPU, cached in `cache/dets/`), `run_track.py` (CPU), `run_eval.py`.
Each run writes `results/<run_id>/` with the resolved `config.yaml`, `env.json`, `tracks/`, `metrics.json`,
`metrics_per_seq.csv` and TrackEval outputs. Configs can inherit from another one with `base:`.

### E1 — reproduction table
```bash
uv run python scripts/run_many.py --configs "configs/e1_*.yaml" --table e1
```
Runs the official setting and the variants below, then writes `results/tables/e1.{csv,md,tex}`.
Only `e1_fp32` needs a second GPU pass; the others reuse the cached detections.

| Config | What changes |
|---|---|
| `e1_default` | official `tools/track.py` settings (match 0.9, no interpolation) |
| `e1_match08` | matching threshold 0.8, as implied by the paper text |
| `e1_interp20` | + linear tracklet interpolation, interval 20 |
| `e1_match08_interp20` | both of the above |
| `e1_sort` | ByteTrack's SORT re-implementation on the same detections |
| `e1_fp32` | detector in FP32 instead of FP16 |

### E2 — static vs moving camera
```bash
uv run python scripts/analyze_e2.py --run e1_default --gap-runs e1_default e1_fp32 e1_interp20 e1_match08 e1_match08_interp20 e1_sort
```
Writes group/sequence tables (`.csv`, `.tex`) and figures to `results/e2/`.

### E3 — camera motion vs Kalman prediction
```bash
uv run python scripts/run_camera_motion.py --config configs/e1_default.yaml   # ORB + RANSAC per frame (CPU)
uv run python scripts/run_e3_log.py --config configs/e1_default.yaml          # read-only association log + per-frame events
uv run python scripts/analyze_e3.py --case-seq MOT17-13-FRCNN                 # tables + figures in results/e3/
```

### E4 — sensitivity study (one parameter at a time)
```bash
uv run python scripts/make_e4_configs.py
uv run python scripts/run_many.py --configs "configs/e4/*.yaml" --table e4
uv run python scripts/analyze_e4.py
```

### Demo (one sequence, about one minute)
```bash
uv run python scripts/demo.py              # MOT17-13: detect (GPU) -> track -> evaluate -> side-by-side video in demo_out/
uv run python scripts/demo.py --cached     # same, with cached detections (no GPU)
```

### Paper assets and project page
```bash
uv run python scripts/make_paper_assets.py   # paper/numbers.tex, paper/tables/, paper/figures/ from results/
uv run python scripts/make_site_data.py      # site/data/results.js for the project page
```
The project page (`site/`) is a static page deployed with GitHub Pages (GitHub Actions, path `site`).

## Credits
- ByteTrack (MIT) — Zhang et al., ECCV 2022. Used unmodified from `external/ByteTrack`; see `src/bt_compat.py` for the compatibility shims. `src/postprocess.py` re-implements its `tools/interpolation.py`.
- TrackEval (MIT) — Luiten et al.
- MOT17 — Milan et al., MOTChallenge (data not redistributed).

## License
MIT (this repository's own code). Third-party code keeps its own license.
