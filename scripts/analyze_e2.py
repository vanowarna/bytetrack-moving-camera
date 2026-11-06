"""E2: static vs moving camera — group metrics, error shares, confounds, figures.

    uv run python scripts/analyze_e2.py --run e1_default --gap-runs e1_default e1_fp32 e1_interp20 e1_match08 e1_match08_interp20 e1_sort

Reads only results/<run>/metrics.json and the val-half ground truth. Writes results/e2/:
    e2_groups.csv/.tex        group metrics + errors per 1,000 GT boxes + share of all errors   (task 4.1, 4.2)
    e2_sequences.csv/.tex     per-sequence metrics next to sequence properties (confounds)     (task 4.3)
    e2_gap_by_setting.csv/.tex static-minus-moving gap for every E1 setting
    figures/fig_e2_per_seq.{pdf,png}, fig_e2_share.{pdf,png}, fig_e2_gap.{pdf,png}            (task 4.4)
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.config import GROUP_COLORS, GROUPS, RESULTS_DIR, SEQ_META, SEQS, VAL_HALF_DIR  # noqa: E402
from src.data.half_split import read_seqinfo  # noqa: E402

OUT = RESULTS_DIR / "e2"
FIG = OUT / "figures"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "savefig.bbox": "tight", "savefig.dpi": 200, "pdf.fonttype": 42})


def load(run):
    return json.loads((RESULTS_DIR / run / "metrics.json").read_text(encoding="utf-8"))


def group_of(seq):
    return next(g for g, m in GROUPS.items() if seq in m)


def seq_properties(seq):
    """Confounds measured from the val-half ground truth (considered pedestrians, class 1, flag 1)."""
    info = read_seqinfo(VAL_HALF_DIR / seq)
    gt = np.loadtxt(VAL_HALF_DIR / seq / "gt" / "gt.txt", delimiter=",")
    ped = gt[(gt[:, 6] == 1) & (gt[:, 7] == 1)]
    n_frames, h_img = int(info["seqLength"]), int(info["imHeight"])
    return {
        "fps": int(info["frameRate"]), "resolution": f"{info['imWidth']}x{info['imHeight']}",
        "frames": n_frames, "gt_boxes": len(ped), "tracks": len(np.unique(ped[:, 1])),
        "density": len(ped) / n_frames,                               # people per frame
        "box_h_rel": float(np.median(ped[:, 5]) / h_img * 100),        # median person height, % of image height
        "visibility": float(np.mean(ped[:, 8])),                       # mean annotated visibility
        **SEQ_META[seq],
    }


def write_csv(path, rows, cols):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_tex(path, rows, cols, headers, fmt=None):
    fmt = fmt or {}
    esc = lambda s: str(s).replace("_", r"\_").replace("%", r"\%")  # noqa: E731
    lines = [r"\begin{tabular}{l" + "r" * (len(cols) - 1) + "}", r"\toprule", " & ".join(headers) + r" \\", r"\midrule"]
    for r in rows:
        cells = []
        for c in cols:
            v = r[c]
            cells.append(esc(fmt[c].format(v) if c in fmt and isinstance(v, (int, float)) else v))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="e1_default")
    ap.add_argument("--gap-runs", nargs="*", default=[])
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    m = load(args.run)
    te, per_seq = m["trackeval"], m["trackeval_per_seq"]

    # ---- 4.1 + 4.2: group metrics, rates and shares --------------------------------------
    tot = {k: sum(te[g][k] for g in GROUPS) for k in ("TP", "FN", "FP", "IDSW")}
    tot["GT"] = tot["TP"] + tot["FN"]
    groups = []
    for g in GROUPS:
        r = te[g]
        gt = r["TP"] + r["FN"]
        groups.append({
            "group": g, "sequences": len(GROUPS[g]), "GT": gt,
            "HOTA": r["HOTA"], "DetA": r["DetA"], "AssA": r["AssA"], "MOTA": r["MOTA"], "IDF1": r["IDF1"],
            "IDSW": r["IDSW"], "FN": r["FN"], "FP": r["FP"],
            "IDSW_per_1k": 1000 * r["IDSW"] / gt, "FN_per_1k": 1000 * r["FN"] / gt, "FP_per_1k": 1000 * r["FP"] / gt,
            "share_GT": 100 * gt / tot["GT"], "share_IDSW": 100 * r["IDSW"] / tot["IDSW"],
            "share_FN": 100 * r["FN"] / tot["FN"], "share_FP": 100 * r["FP"] / tot["FP"],
        })
    gcols = list(groups[0].keys())
    write_csv(OUT / "e2_groups.csv", groups, gcols)
    write_tex(OUT / "e2_groups.tex", groups,
              ["group", "GT", "HOTA", "DetA", "AssA", "IDF1", "IDSW_per_1k", "FN_per_1k", "share_GT", "share_IDSW"],
              ["Camera", "GT boxes", "HOTA", "DetA", "AssA", "IDF1", "IDSW/1k", "FN/1k", "GT \\%", "IDSW \\%"],
              {k: "{:.1f}" for k in ["HOTA", "DetA", "AssA", "IDF1", "IDSW_per_1k", "FN_per_1k", "share_GT", "share_IDSW"]})

    # ---- 4.3: per-sequence metrics next to confounds --------------------------------------
    seqs = []
    for s in SEQS:
        r, p = per_seq[s], seq_properties(s)
        gt = r["TP"] + r["FN"]
        seqs.append({"seq": s.replace("-FRCNN", ""), "group": group_of(s), **p,
                     "HOTA": r["HOTA"], "DetA": r["DetA"], "AssA": r["AssA"], "MOTA": r["MOTA"], "IDF1": r["IDF1"],
                     "IDSW": r["IDSW"], "IDSW_per_1k": 1000 * r["IDSW"] / gt})
    seqs.sort(key=lambda r: (r["group"] != "static", r["seq"]))
    write_csv(OUT / "e2_sequences.csv", seqs, list(seqs[0].keys()))
    write_tex(OUT / "e2_sequences.tex", seqs,
              ["seq", "camera", "fps", "density", "box_h_rel", "conditions", "HOTA", "DetA", "AssA", "IDSW_per_1k"],
              ["Seq.", "Camera", "FPS", "People/frame", "Height \\%", "Scene", "HOTA", "DetA", "AssA", "IDSW/1k"],
              {"density": "{:.1f}", "box_h_rel": "{:.1f}", "HOTA": "{:.1f}", "DetA": "{:.1f}", "AssA": "{:.1f}",
               "IDSW_per_1k": "{:.1f}"})

    # ---- gap per E1 setting ------------------------------------------------------------------
    gaps = []
    for run in args.gap_runs:
        t = load(run)["trackeval"]
        cfg_note = ""
        cfgp = RESULTS_DIR / run / "config.yaml"
        if cfgp.exists():
            import yaml
            cfg_note = (yaml.safe_load(cfgp.read_text(encoding="utf-8")) or {}).get("note", "")
        row = {"run": run, "note": cfg_note}
        for k in ("HOTA", "DetA", "AssA", "IDF1"):
            row[f"{k}_static"], row[f"{k}_moving"] = t["static"][k], t["moving"][k]
            row[f"{k}_gap"] = t["static"][k] - t["moving"][k]
        gaps.append(row)
    if gaps:
        write_csv(OUT / "e2_gap_by_setting.csv", gaps, list(gaps[0].keys()))
        write_tex(OUT / "e2_gap_by_setting.tex", gaps,
                  ["note", "HOTA_static", "HOTA_moving", "HOTA_gap", "DetA_gap", "AssA_gap", "IDF1_gap"],
                  ["Setting", "HOTA static", "HOTA moving", "$\\Delta$HOTA", "$\\Delta$DetA", "$\\Delta$AssA", "$\\Delta$IDF1"],
                  {k: "{:.1f}" for k in ["HOTA_static", "HOTA_moving", "HOTA_gap", "DetA_gap", "AssA_gap", "IDF1_gap"]})

    # ---- 4.4 figures -------------------------------------------------------------------------
    # (a) per-sequence HOTA / DetA / AssA, bars coloured by camera group
    fig, ax = plt.subplots(figsize=(6.8, 2.6))
    x = np.arange(len(seqs))
    for i, (k, hatch) in enumerate([("HOTA", ""), ("DetA", "///"), ("AssA", "...")]):
        ax.bar(x + (i - 1) * 0.26, [r[k] for r in seqs], 0.24, color=[GROUP_COLORS[r["group"]] for r in seqs],
               hatch=hatch, edgecolor="white", linewidth=0.4, label=k)
    ax.set_xticks(x, [r["seq"].replace("MOT17-", "") for r in seqs])
    ax.set_xlabel("MOT17 sequence (val half)")
    ax.set_ylabel("score")
    ax.set_ylim(30, 95)
    split = sum(r["group"] == "static" for r in seqs) - 0.5
    ax.axvline(split, color="#888780", lw=0.8, ls="--")
    ax.text(split - 0.1, 90, "static", ha="right", color=GROUP_COLORS["static"])
    ax.text(split + 0.1, 90, "moving", ha="left", color=GROUP_COLORS["moving"])
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor="#b4b2a9", hatch=h, edgecolor="white") for h in ("", "///", "...")]
    ax.legend(handles, ["HOTA", "DetA", "AssA"], ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    save(fig, "fig_e2_per_seq")

    # (b) share of GT boxes vs share of each error type
    fig, ax = plt.subplots(figsize=(4.2, 2.2))
    cats = ["GT boxes", "FN", "FP", "ID switches"]
    keys = ["share_GT", "share_FN", "share_FP", "share_IDSW"]
    left = np.zeros(len(cats))
    for g in groups:
        vals = np.array([g[k] for k in keys])
        ax.barh(cats, vals, left=left, color=GROUP_COLORS[g["group"]], edgecolor="white", label=g["group"])
        for i, v in enumerate(vals):
            ax.text(left[i] + v / 2, i, f"{v:.0f}%", ha="center", va="center", color="white", fontsize=8)
        left += vals
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of total (%)")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.25))
    save(fig, "fig_e2_share")

    # (c) HOTA static vs moving under every E1 setting
    if gaps:
        fig, ax = plt.subplots(figsize=(4.6, 2.4))
        y = np.arange(len(gaps))
        for r, yy in zip(gaps, y):
            ax.plot([r["HOTA_moving"], r["HOTA_static"]], [yy, yy], color="#b4b2a9", lw=2, zorder=1)
        ax.scatter([r["HOTA_static"] for r in gaps], y, color=GROUP_COLORS["static"], zorder=2, label="static", s=28)
        ax.scatter([r["HOTA_moving"] for r in gaps], y, color=GROUP_COLORS["moving"], zorder=2, label="moving", s=28,
                   marker="D")
        ax.set_yticks(y, [(r["note"] or r["run"]).split(" (")[0] for r in gaps])
        ax.invert_yaxis()
        ax.set_xlabel("HOTA")
        ax.grid(axis="y", visible=False)
        ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.2))
        save(fig, "fig_e2_gap")

    # ---- console summary ---------------------------------------------------------------------
    for g in groups:
        print(f"{g['group']:7s} GT {g['share_GT']:4.1f}% | IDSW {g['share_IDSW']:4.1f}% | FN {g['share_FN']:4.1f}% | "
              f"IDSW/1k {g['IDSW_per_1k']:.2f} | HOTA {g['HOTA']:.1f} DetA {g['DetA']:.1f} AssA {g['AssA']:.1f}")
    for r in gaps:
        print(f"gap {r['run']:22s} HOTA {r['HOTA_gap']:5.1f}  DetA {r['DetA_gap']:5.1f}  AssA {r['AssA_gap']:5.1f}  IDF1 {r['IDF1_gap']:5.1f}")
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
