"""Run several configs through the pipeline, then build one comparison table.

    uv run python scripts/run_many.py --configs "configs/e1_*.yaml" --table e1
    uv run python scripts/run_many.py --configs "configs/e1_*.yaml" --table e1 --skip "e1_fp32"

Detection is cached, so configs that share a detector reuse it (only tracking + evaluation run).
"""
import argparse
import glob
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True, help="paths or glob patterns")
    ap.add_argument("--table", required=True, help="table name -> results/tables/<name>.*")
    ap.add_argument("--skip", nargs="*", default=[], help="run_ids to skip")
    args = ap.parse_args()

    paths = []
    for pat in args.configs:
        paths += sorted(glob.glob(pat)) or [pat]
    run_ids = []
    for p in paths:
        cfg = load_config(p)
        if cfg["run_id"] in args.skip:
            continue
        print(f"\n===== {cfg['run_id']}  ({p}) =====", flush=True)
        subprocess.run([sys.executable, str(HERE / "run_pipeline.py"), "--config", p], check=True)
        run_ids.append(cfg["run_id"])
    # default run first in the table
    run_ids.sort(key=lambda r: (not r.endswith("_default"), r))
    subprocess.run([sys.executable, str(HERE / "make_table.py"), "--name", args.table, "--runs", *run_ids], check=True)


if __name__ == "__main__":
    main()
