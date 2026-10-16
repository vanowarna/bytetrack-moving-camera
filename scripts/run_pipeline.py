"""Detect (cached) -> track -> evaluate, in one command.

    uv run python scripts/run_pipeline.py --config configs/p1_smoke.yaml
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(script: str, cfg: str, *extra: str):
    cmd = [sys.executable, str(HERE / script), "--config", cfg, *extra]
    print(f"\n$ {' '.join(cmd[1:])}")
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--skip-detect", action="store_true")
    args = ap.parse_args()
    if not args.skip_detect:
        run("run_detect.py", args.config)
    run("run_track.py", args.config)
    run("run_eval.py", args.config)


if __name__ == "__main__":
    main()
