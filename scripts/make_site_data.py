"""Write site/data/results.js for the project page from the files in results/.

Every number on the project page is read from results/, as for the paper (make_paper_assets.py).
Standard library only.

    uv run python scripts/make_site_data.py
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "site" / "data" / "results.js"


def rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(x, nd=None):
    if x is None or x == "":
        return None
    try:
        v = float(x)
    except ValueError:
        return x
    return round(v, nd) if nd is not None else v


def main():
    data = {}

    # E1: reproduction table
    data["e1"] = [
        {
            "run": r["run"], "note": r["note"],
            "MOTA": num(r["MOTA*"], 1), "IDF1": num(r["IDF1*"], 1), "IDs": num(r["IDs*"], 0),
            "HOTA": num(r["HOTA"], 1), "DetA": num(r["DetA"], 1), "AssA": num(r["AssA"], 1),
            "HOTA_static": num(r["HOTA static"], 1), "HOTA_moving": num(r["HOTA moving"], 1),
        }
        for r in rows(RES / "tables" / "e1.csv")
    ]

    # E2: groups, sequences, gap per setting
    data["groups"] = {
        r["group"]: {k: num(r[k], 2) for k in
                     ["GT", "HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "IDSW_per_1k",
                      "share_GT", "share_IDSW", "share_FN", "share_FP"]}
        for r in rows(RES / "e2" / "e2_groups.csv")
    }
    e3seq = {r["seq"]: r for r in rows(RES / "e3" / "e3_per_seq.csv")}
    seqs = []
    for r in rows(RES / "e2" / "e2_sequences.csv"):
        short = r["seq"].split("-")[1]
        m = e3seq.get(short, {})
        seqs.append({
            "seq": r["seq"], "group": r["group"], "fps": num(r["fps"], 0), "resolution": r["resolution"],
            "frames": num(r["frames"], 0), "gt": num(r["gt_boxes"], 0),
            "density": num(r["density"], 1), "height": num(r["box_h_rel"], 1),
            "visibility": num(r["visibility"], 2), "viewpoint": r["viewpoint"], "conditions": r["conditions"],
            "HOTA": num(r["HOTA"], 1), "DetA": num(r["DetA"], 1), "AssA": num(r["AssA"], 1),
            "MOTA": num(r["MOTA"], 1), "IDF1": num(r["IDF1"], 1), "IDSW": num(r["IDSW"], 0),
            "IDSW_per_1k": num(r["IDSW_per_1k"], 1),
            "motion": num(m.get("motion_median"), 3), "pred_iou": num(m.get("pred_iou"), 2),
            "rho": num(m.get("rho_motion_iou"), 2), "fit_ok": num(m.get("fit_ok_pct"), 1),
        })
    data["sequences"] = seqs
    data["gap_by_setting"] = [
        {"run": r["run"], "note": r["note"],
         "HOTA": num(r["HOTA_gap"], 1), "DetA": num(r["DetA_gap"], 1), "AssA": num(r["AssA_gap"], 1)}
        for r in rows(RES / "e2" / "e2_gap_by_setting.csv")
    ]

    # E3: summary, lost tracks, per-frame timeline
    s = json.loads((RES / "e3" / "e3_summary.json").read_text(encoding="utf-8"))
    data["e3"] = {
        "motion_static_max": round(max(s["motion_median_static"]), 3),
        "motion_moving": [round(v, 2) for v in s["motion_median_moving"]],
        "rho_moving": [round(v, 2) for v in s["rho_motion_iou_moving"]],
        "err_low": round(s["err_low_flow_moving"], 3), "err_high": round(s["err_high_flow_moving"], 3),
        "rho_flow_err": round(s["rho_flow_err"], 2), "n_matches": s["n_matches"],
        "fit_ok_min": int(s["fit_ok_min"]),
        "high": {"switch": s["high_motion"]["switch"], "per_1k": round(s["high_motion"]["switch_per_1k"], 2),
                 "iou": round(s["high_motion"]["iou"], 2)},
        "low": {"switch": s["low_motion"]["switch"], "per_1k": round(s["low_motion"]["switch_per_1k"], 2),
                "iou": round(s["low_motion"]["iou"], 2)},
        "lost_rho_within": round(s["lost_ctrl"]["rho_within_gap_median"], 2),
        "lost_rho_pooled": round(s["lost_ctrl"]["rho_pooled"], 2),
    }
    data["lost"] = [
        {"group": r["group"], "gap": r["gap"], "n": num(r["n"], 0),
         "refound": num(r["rematch_pct"], 1), "best_iou": num(r["best_iou"], 2)}
        for r in rows(RES / "e3" / "e3_lost.csv")
    ]
    timeline = {}
    for r in rows(RES / "e3" / "e3_frames.csv"):
        t = timeline.setdefault(r["seq"], {"group": r["group"], "f": [], "m": [], "iou": [], "sw": []})
        t["f"].append(int(r["frame"]))
        t["m"].append(num(r["disp_pct"], 3))
        t["iou"].append(num(r["iou"], 3))
        t["sw"].append(int(float(r["switch"] or 0)))
    data["timeline"] = timeline

    # E4: sensitivity
    data["e4"] = [
        {"param": r["param"], "value": num(r["value"], 2), "official": r["run"] == "e1_default",
         "HOTA": num(r["HOTA"], 1), "static": num(r["HOTA_static"], 1), "moving": num(r["HOTA_moving"], 1),
         "gap": num(r["gap_HOTA"], 1), "gap_AssA": num(r["gap_AssA"], 1), "gap_DetA": num(r["gap_DetA"], 1)}
        for r in rows(RES / "e4" / "e4_sensitivity.csv")
    ]

    # Environment of the official run (no local paths)
    env = json.loads((RES / "e1_default" / "env.json").read_text(encoding="utf-8"))
    data["env"] = {k: env.get(k) for k in ["python", "cuda", "gpu", "bytetrack_commit", "trackeval_commit"]}
    data["env"]["torch"] = env.get("packages", {}).get("torch")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("// generated by scripts/make_site_data.py from results/ - do not edit\n"
                   "window.RESULTS = " + json.dumps(data, separators=(",", ":")) + ";\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    sys.exit(main())
