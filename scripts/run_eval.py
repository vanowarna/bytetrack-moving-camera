"""Evaluate a run: TrackEval (HOTA/DetA/AssA/MOTA/IDF1/IDSW) for all sequences and per camera group,
plus py-motmetrics MOTA/IDF1 exactly as ByteTrack computes them.

    uv run python scripts/run_eval.py --config configs/p1_smoke.yaml

Output: results/<run_id>/metrics.json, metrics_per_seq.csv, trackeval/
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tabulate import tabulate  # noqa: E402

from src.config import GROUPS, load_config, run_dir  # noqa: E402
from src.eval import motmetrics_runner, trackeval_runner  # noqa: E402
from src.runlog import write_json  # noqa: E402

COLS = ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "FP", "FN"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = run_dir(cfg)
    seqs = cfg["seqs"]

    te = {"all": trackeval_runner.evaluate(out, seqs, name="all")}
    for g, members in GROUPS.items():
        sub = [s for s in seqs if s in members]
        if sub and len(sub) < len(seqs):
            te[g] = trackeval_runner.evaluate(out, sub, name=g)
    mm = motmetrics_runner.evaluate(out, seqs)

    metrics = {
        "run_id": cfg["run_id"],
        "seqs": seqs,
        "trackeval": {k: v["combined"] for k, v in te.items()},
        "trackeval_per_seq": te["all"]["per_seq"],
        "motmetrics_bytetrack_protocol": mm,
    }
    write_json(out / "metrics.json", metrics)

    with open(out / "metrics_per_seq.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seq", "group", *COLS, "MOTA_mm", "IDF1_mm"])
        for seq, r in te["all"]["per_seq"].items():
            grp = next((g for g, m in GROUPS.items() if seq in m), "")
            w.writerow([seq, grp, *[round(r[c], 3) for c in COLS], round(mm[seq]["mota"], 3), round(mm[seq]["idf1"], 3)])

    rows = [[seq, *[f"{r[c]:.1f}" if isinstance(r[c], float) else r[c] for c in COLS]]
            for seq, r in te["all"]["per_seq"].items()]
    for k, v in metrics["trackeval"].items():
        rows.append([f"[{k}]", *[f"{v[c]:.1f}" if isinstance(v[c], float) else v[c] for c in COLS]])
    print(tabulate(rows, headers=["seq", *COLS]))
    c = mm["combined"]
    print(f"\nByteTrack protocol (motmetrics): MOTA {c['mota']:.1f}  IDF1 {c['idf1']:.1f}  "
          f"IDs {int(c['num_switches'])}   [paper, full val half: 76.6 / 79.3 / 159]")
    print(f"Saved {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
