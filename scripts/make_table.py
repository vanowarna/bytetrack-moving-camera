"""Collect metrics.json of several runs into one table (CSV, Markdown and a LaTeX tabular for the paper).

    uv run python scripts/make_table.py --name e1 --runs e1_default e1_match08 e1_interp20 e1_sort

Output: results/tables/<name>.csv, <name>.md, <name>.tex
Every number comes from results/<run_id>/metrics.json. The paper row is the published reference only.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from src.config import RESULTS_DIR  # noqa: E402

PAPER = {"run": "paper", "note": "ByteTrack paper (published)", "MOTA*": 76.6, "IDF1*": 79.3, "IDs*": 159}
COLS = ["run", "note", "MOTA*", "IDF1*", "IDs*", "HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW",
        "HOTA static", "HOTA moving"]


def row_for(run_id: str) -> dict:
    d = RESULTS_DIR / run_id
    m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    note = ""
    if (d / "config.yaml").exists():
        note = (yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8")) or {}).get("note", "")
    mm, te = m["motmetrics_bytetrack_protocol"]["combined"], m["trackeval"]
    a = te["all"]
    return {
        "run": run_id, "note": note,
        "MOTA*": round(mm["mota"], 1), "IDF1*": round(mm["idf1"], 1), "IDs*": int(mm["num_switches"]),
        "HOTA": round(a["HOTA"], 1), "DetA": round(a["DetA"], 1), "AssA": round(a["AssA"], 1),
        "MOTA": round(a["MOTA"], 1), "IDF1": round(a["IDF1"], 1), "IDSW": a["IDSW"],
        "HOTA static": round(te["static"]["HOTA"], 1) if "static" in te else "",
        "HOTA moving": round(te["moving"]["HOTA"], 1) if "moving" in te else "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--no-paper-row", action="store_true")
    args = ap.parse_args()

    rows = ([] if args.no_paper_row else [PAPER]) + [row_for(r) for r in args.runs]
    out = RESULTS_DIR / "tables"
    out.mkdir(parents=True, exist_ok=True)

    with open(out / f"{args.name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLS})

    md = ["| " + " | ".join(COLS) + " |", "|" + "---|" * len(COLS)]
    md += ["| " + " | ".join(str(r.get(c, "")) for c in COLS) + " |" for r in rows]
    md.append("\n`*` = py-motmetrics as in ByteTrack's own script; other columns = TrackEval.")
    (out / f"{args.name}.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    tex_cols = [c for c in COLS if c != "run"]
    esc = lambda s: str(s).replace("_", r"\_").replace("%", r"\%")  # noqa: E731
    tex = [r"\begin{tabular}{l" + "r" * (len(tex_cols) - 1) + "}", r"\toprule",
           " & ".join(["Setting"] + [esc(c) for c in tex_cols[1:]]) + r" \\", r"\midrule"]
    for r in rows:
        tex.append(" & ".join([esc(r.get("note") or r["run"])] + [esc(r.get(c, "")) for c in tex_cols[1:]]) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (out / f"{args.name}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

    print("\n".join(md))
    print(f"\nSaved {out / args.name}.csv / .md / .tex")


if __name__ == "__main__":
    main()
