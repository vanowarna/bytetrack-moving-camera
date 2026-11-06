"""Build every number, table and figure used by the paper from results/ (nothing is typed by hand).

    uv run python scripts/make_paper_assets.py

Writes
    paper/numbers.tex          \\newcommand macros for every number quoted in the text
    paper/tables/tab_*.tex     tables
    paper/figures/fig_*.pdf    copies of the analysis figures (PNG if no PDF exists)
Re-run it after any experiment changes, then recompile the paper.
"""
import csv
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RESULTS_DIR, ROOT  # noqa: E402

PAPER = ROOT / "paper"
TAB, FIG = PAPER / "tables", PAPER / "figures"

E1_RUNS = [  # (run_id, label in the table)
    ("e1_default", "Official settings"),
    ("e1_interp20", "+ interpolation (20)"),
    ("e1_match08", "Match thr. 0.8"),
    ("e1_match08_interp20", "Match 0.8 + interp."),
    ("e1_fp32", "Detector in FP32"),
    ("e1_sort", "SORT, same detections"),
]
FIGURES = ["fig_e4_sensitivity", "fig_e3_case", "fig_e2_per_seq", "fig_e2_gap", "fig_e3_binned", "fig_e3_error", "fig_e3_lost", "fig_e3_timeline"]


def jload(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def rcsv(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f1(x):
    return f"{float(x):.1f}"


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    N = {}  # macro name -> value string

    # ---------------- E1 ----------------
    e1 = {r: jload(RESULTS_DIR / r / "metrics.json") for r, _ in E1_RUNS}
    rows = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
            r"Setting & MOTA$^*$ & IDF1$^*$ & IDs$^*$ & HOTA & DetA & AssA \\", r"\midrule",
            r"Paper~\cite{zhang2022bytetrack} & 76.6 & 79.3 & 159 & -- & -- & -- \\", r"\midrule"]
    for run, label in E1_RUNS:
        mm, te = e1[run]["motmetrics_bytetrack_protocol"]["combined"], e1[run]["trackeval"]["all"]
        rows.append(f"{label} & {f1(mm['mota'])} & {f1(mm['idf1'])} & {int(mm['num_switches'])} & "
                    f"{f1(te['HOTA'])} & {f1(te['DetA'])} & {f1(te['AssA'])} \\\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_e1.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    d = e1["e1_default"]
    mm, te = d["motmetrics_bytetrack_protocol"]["combined"], d["trackeval"]
    N.update(EoneMOTA=f1(mm["mota"]), EoneIDF=f1(mm["idf1"]), EoneIDs=int(mm["num_switches"]),
             EoneHOTA=f1(te["all"]["HOTA"]), EoneDetA=f1(te["all"]["DetA"]), EoneAssA=f1(te["all"]["AssA"]))
    i = e1["e1_interp20"]["motmetrics_bytetrack_protocol"]["combined"]
    N.update(InterpMOTA=f1(i["mota"]), InterpIDF=f1(i["idf1"]))
    m8 = e1["e1_match08"]["motmetrics_bytetrack_protocol"]["combined"]
    N.update(MatchEightMOTA=f1(m8["mota"]), MatchEightIDF=f1(m8["idf1"]), MatchEightIDs=int(m8["num_switches"]))
    fp = e1["e1_fp32"]["motmetrics_bytetrack_protocol"]["combined"]
    N.update(FpThirtyTwoMOTA=f1(fp["mota"]), FpThirtyTwoIDF=f1(fp["idf1"]))
    so = e1["e1_sort"]["motmetrics_bytetrack_protocol"]["combined"]
    N.update(SortMOTA=f1(so["mota"]), SortIDF=f1(so["idf1"]), SortIDs=int(so["num_switches"]))
    for key, did in (("DetMsFP", "yolox_x_ablation_800x1440_fp16"), ("DetMsFPthirtytwo", "yolox_x_ablation_800x1440_fp32")):
        meta = ROOT / "cache" / "dets" / did / "meta.json"
        N[key] = "n/a"
        if meta.exists():
            t = jload(meta)["timing"]
            n = sum(v["frames"] for v in t.values())
            N[key] = f"{sum(v['ms_mean'] * v['frames'] for v in t.values()) / n:.0f}"

    # ---------------- E2 ----------------
    groups = {r["group"]: r for r in rcsv(RESULTS_DIR / "e2" / "e2_groups.csv")}
    for g, G in (("static", "Static"), ("moving", "Moving")):
        r = groups[g]
        N.update({f"{G}HOTA": f1(r["HOTA"]), f"{G}DetA": f1(r["DetA"]), f"{G}AssA": f1(r["AssA"]),
                  f"{G}IDF": f1(r["IDF1"]), f"{G}ShareGT": f"{float(r['share_GT']):.0f}",
                  f"{G}ShareIDSW": f"{float(r['share_IDSW']):.0f}", f"{G}IDSWk": f"{float(r['IDSW_per_1k']):.1f}"})
    N["GapHOTA"] = f"{float(groups['static']['HOTA']) - float(groups['moving']['HOTA']):.1f}"
    N["GapDetA"] = f"{float(groups['static']['DetA']) - float(groups['moving']['DetA']):.1f}"
    N["GapAssA"] = f"{float(groups['static']['AssA']) - float(groups['moving']['AssA']):.1f}"
    gaps = [float(r["HOTA_gap"]) for r in rcsv(RESULTS_DIR / "e2" / "e2_gap_by_setting.csv")]
    N.update(GapMin=f"{min(gaps):.1f}", GapMax=f"{max(gaps):.1f}")

    # ---------------- E3 ----------------
    s = jload(RESULTS_DIR / "e3" / "e3_summary.json")
    N.update(MotionStaticMax=f"{s['motion_median_static'][1]:.3f}",
             MotionMovingMin=f"{s['motion_median_moving'][0]:.2f}", MotionMovingMax=f"{s['motion_median_moving'][1]:.2f}",
             RhoMovMin=f"{s['rho_motion_iou_moving'][0]:.2f}", RhoMovMax=f"{s['rho_motion_iou_moving'][1]:.2f}",
             RhoStatMin=f"{s['rho_motion_iou_static'][0]:.2f}", RhoStatMax=f"{s['rho_motion_iou_static'][1]:.2f}",
             ErrLow=f"{s['err_low_flow_moving']:.3f}", ErrHigh=f"{s['err_high_flow_moving']:.3f}",
             RhoFlowErr=f"{s['rho_flow_err']:.2f}", NMatches=f"{s['n_matches']:,}".replace(",", "{,}"),
             FitOkMin=f"{s['fit_ok_min']:.0f}",
             LostStatic=f"{s['lost_rematch_9_16']['static']:.1f}", LostMoving=f"{s['lost_rematch_9_16']['moving']:.1f}",
             LostIoUStatic=f"{s['lost_bestiou_9_16']['static']:.2f}", LostIoUMoving=f"{s['lost_bestiou_9_16']['moving']:.2f}",
             LostRhoWithin=f"{s['lost_ctrl']['rho_within_gap_median']:.2f}".replace("-0.00", "0.00"), LostRhoPooled=f"{s['lost_ctrl']['rho_pooled']:.2f}",
             HighSw=s["high_motion"]["switch"], LowSw=s["low_motion"]["switch"],
             HighSwk=f"{s['high_motion']['switch_per_1k']:.2f}", LowSwk=f"{s['low_motion']['switch_per_1k']:.2f}",
             HighIoU=f"{s['high_motion']['iou']:.2f}", LowIoU=f"{s['low_motion']['iou']:.2f}")

    # combined per-sequence table (E2 properties + metrics, E3 motion + prediction)
    e2s = {r["seq"].replace("MOT17-", ""): r for r in rcsv(RESULTS_DIR / "e2" / "e2_sequences.csv")}
    e3s = {r["seq"]: r for r in rcsv(RESULTS_DIR / "e3" / "e3_per_seq.csv")}
    rows = [r"\begin{tabular}{llrrrlrrrrrrr}", r"\toprule",
            r" & & & People/ & Height & & & & & IDSW/ & Motion & Pred. & $\rho$(motion, \\",
            r"Seq. & Cam. & FPS & frame & (\%) & Scene & HOTA & DetA & AssA & 1k GT & (\%diag) & IoU & pred.\ IoU) \\",
            r"\midrule"]
    order = sorted(e2s, key=lambda k: (e2s[k]["group"] != "static", k))
    for k in order:
        a, b = e2s[k], e3s[k]
        rows.append(f"{k} & {a['camera'][0].upper()} & {a['fps']} & {float(a['density']):.1f} & {float(a['box_h_rel']):.0f} & "
                    f"{a['conditions']} & {f1(a['HOTA'])} & {f1(a['DetA'])} & {f1(a['AssA'])} & {float(a['IDSW_per_1k']):.1f} & "
                    f"{float(b['motion_median']):.3f} & {float(b['pred_iou']):.2f} & {float(b['rho_motion_iou']):+.2f} \\\\")
        if k == order[2]:
            rows.append(r"\midrule")
    rows += [r"\midrule"]
    for g in ("static", "moving"):
        r = groups[g]
        rows.append(f"\\multicolumn{{6}}{{l}}{{All {g}}} & {f1(r['HOTA'])} & {f1(r['DetA'])} & {f1(r['AssA'])} & "
                    f"{float(r['IDSW_per_1k']):.1f} & & & \\\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_seq.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    lost = rcsv(RESULTS_DIR / "e3" / "e3_lost.csv")
    by = {(r["group"], r["gap"]): r for r in lost}
    gaps_order = ["2", "3-4", "5-8", "9-16", "17-30"]
    rows = [r"\begin{tabular}{lrrrr}", r"\toprule",
            r"Frames & \multicolumn{2}{c}{Re-found (\%)} & \multicolumn{2}{c}{Best IoU} \\",
            r"lost & static & moving & static & moving \\", r"\midrule"]
    for gp in gaps_order:
        st, mv = by[("static", gp)], by[("moving", gp)]
        rows.append(f"{gp} & {float(st['rematch_pct']):.1f} & {float(mv['rematch_pct']):.1f} & "
                    f"{float(st['best_iou']):.2f} & {float(mv['best_iou']):.2f} \\\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_lost.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    # ---------------- E4 ----------------
    e4p = RESULTS_DIR / "e4" / "e4_sensitivity.csv"
    if e4p.exists():
        e4 = rcsv(e4p)
        sm = jload(RESULTS_DIR / "e4" / "e4_summary.json")
        N.update(EfourN=sm["n_settings"], EfourGapMin=f"{sm['gap_min']:.1f}", EfourGapMax=f"{sm['gap_max']:.1f}",
                 EfourAssaCount=sm["assa_gt_deta_count"])
        other = [r for r in e4 if r["param"] != "kalman_noise_scale"]
        N.update(EfourOtherGapMin=f"{min(float(r['gap_HOTA']) for r in other):.1f}",
                 EfourOtherGapMax=f"{max(float(r['gap_HOTA']) for r in other):.1f}",
                 EfourOtherMovMin=f"{min(float(r['HOTA_moving']) for r in other):.1f}",
                 EfourOtherMovMax=f"{max(float(r['HOTA_moving']) for r in other):.1f}")
        kf = {float(r["value"]): r for r in e4 if r["param"] == "kalman_noise_scale"}
        for v, tag in ((0.25, "Quarter"), (2.0, "Two")):
            r = kf[v]
            N.update({f"Kf{tag}Mov": f1(r["HOTA_moving"]), f"Kf{tag}Stat": f1(r["HOTA_static"]),
                      f"Kf{tag}Gap": f1(r["gap_HOTA"]), f"Kf{tag}HOTA": f1(r["HOTA"])})

    # ---------------- figures ----------------
    missing = []
    for name in FIGURES:
        src = None
        for sub in ("e2", "e3", "e4"):
            for ext in ("pdf", "png"):
                p = RESULTS_DIR / sub / "figures" / f"{name}.{ext}"
                if p.exists() and src is None:
                    src = p
        if src is None:
            missing.append(name)
            continue
        for old in FIG.glob(f"{name}.*"):
            old.unlink()
        shutil.copyfile(src, FIG / src.name)

    # ---------------- numbers.tex ----------------
    lines = ["% AUTO-GENERATED by scripts/make_paper_assets.py from results/ -- do not edit by hand"]
    for k, v in N.items():
        lines.append(f"\\newcommand{{\\v{k}}}{{{v}}}")
    (PAPER / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"{len(N)} numbers -> paper/numbers.tex; tables -> paper/tables/; figures -> paper/figures/")
    if missing:
        print("missing figures:", missing)


if __name__ == "__main__":
    main()
