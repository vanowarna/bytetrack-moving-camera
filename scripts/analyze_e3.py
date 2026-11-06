"""E3 tasks 5.2, 5.4, 5.5: relate camera motion to the Kalman prediction and to tracking errors.

    uv run python scripts/analyze_e3.py
    uv run python scripts/analyze_e3.py --case-seq MOT17-13-FRCNN     # also draws the case figure (needs images)

Inputs (from run_camera_motion.py and run_e3_log.py):
    results/e3/camera_motion/<seq>.csv, results/e3/assoc/<seq>.csv.gz, results/e3/events/<seq>.csv
Definitions
    one-step prediction : a track that was updated in the previous frame (since == 1) and is in the
                          'tracked' state, at BYTE's first (high-score) association
    pred IoU            : IoU between its Kalman-predicted box and the detection it was matched to
    unmatched rate      : share of one-step predictions that found no high-score detection
    pred error          : |centre(pred) - centre(det)| / det height   (matched only)
    camera flow at box  : |A_t c - c| / box height, the image motion the camera causes at the box centre
    camera change       : |(A_t - A_{t-1}) c| / box height, the part a constant-velocity model cannot predict
Outputs: results/e3/e3_*.csv/.tex and results/e3/figures/fig_e3_*.{pdf,png}
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from src.config import GROUP_COLORS, GROUPS, RESULTS_DIR, SEQS  # noqa: E402

E3 = RESULTS_DIR / "e3"
FIG = E3 / "figures"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "savefig.bbox": "tight", "savefig.dpi": 200, "pdf.fonttype": 42})
SEQ_STYLE = {s: ls for s, ls in zip(SEQS, ["-", "--", "-", ":", "--", "-.", ":"])}


def group_of(seq):
    return next(g for g, m in GROUPS.items() if seq in m)


def short(seq):
    return seq.replace("MOT17-", "").replace("-FRCNN", "")


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)


def rho(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10:
        return float("nan"), float("nan")
    r, p = spearmanr(a[ok], b[ok])
    return float(r), float(p)


def load_seq(seq):
    cam = pd.read_csv(E3 / "camera_motion" / f"{seq}.csv")
    log = pd.read_csv(E3 / "assoc" / f"{seq}.csv.gz")
    ev = pd.read_csv(E3 / "events" / f"{seq}.csv")
    one = log[(log.stage == "high") & (log.state == "tracked") & (log.since == 1)].copy()
    one = one.merge(cam[["frame"] + [c for c in cam.columns if c.startswith("a") and len(c) == 3]], on="frame", how="left")
    prev = cam[["frame"] + [c for c in cam.columns if c.startswith("a") and len(c) == 3]].copy()
    prev["frame"] += 1
    one = one.merge(prev, on="frame", how="left", suffixes=("", "_prev"))
    cx, cy = (one.px1 + one.px2) / 2, (one.py1 + one.py2) / 2
    hp = (one.py2 - one.py1).clip(lower=1)
    fx = one.a00 * cx + one.a01 * cy + one.a02 - cx
    fy = one.a10 * cx + one.a11 * cy + one.a12 - cy
    one["cam_flow_rel"] = np.hypot(fx, fy) / hp
    gx = (one.a00 - one.a00_prev) * cx + (one.a01 - one.a01_prev) * cy + (one.a02 - one.a02_prev)
    gy = (one.a10 - one.a10_prev) * cx + (one.a11 - one.a11_prev) * cy + (one.a12 - one.a12_prev)
    one["cam_change_rel"] = np.hypot(gx, gy) / hp
    m = one.matched == 1
    dcx, dcy, hd = (one.dx1 + one.dx2) / 2, (one.dy1 + one.dy2) / 2, (one.dy2 - one.dy1).clip(lower=1)
    one.loc[m, "pred_err"] = np.hypot(cx - dcx, cy - dcy)[m] / hd[m]

    per_frame = one.groupby("frame").agg(n_pred=("track_id", "size"), unmatched=("matched", lambda s: (s == 0).mean()),
                                         iou=("iou", "mean"), pred_err=("pred_err", "median")).reset_index()
    frames = cam[["frame", "ok", "disp_pct", "accel_pct"]].merge(per_frame, on="frame", how="left").merge(ev, on="frame", how="left")
    frames["seq"], frames["group"] = seq, group_of(seq)
    one["seq"], one["group"] = seq, group_of(seq)
    return frames, one


def binned(ax, x, y, bins, color, label, ls="-", marker="o"):
    idx = np.digitize(x, bins[1:-1])
    xs, ys = [], []
    for b in range(len(bins) - 1):
        sel = (idx == b) & np.isfinite(y)
        if sel.sum() >= 20:
            xs.append(np.median(x[sel]))
            ys.append(np.mean(y[sel]))
    ax.plot(xs, ys, ls, color=color, marker=marker, ms=4, lw=1.5, label=label)


def case_figure(seq, frames, one):
    """5.5: the frame with the largest camera change that still has >= 5 one-step predictions."""
    import cv2
    from src.data.half_split import val_frames
    f = frames[(frames.seq == seq) & (frames.n_pred >= 5)].sort_values("accel_pct", ascending=False).iloc[0]
    frame = int(f.frame)
    path = dict(val_frames(seq))[frame]
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    rows = one[(one.seq == seq) & (one.frame == frame)]
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    ax.imshow(img)
    for _, r in rows.iterrows():
        ax.add_patch(plt.Rectangle((r.px1, r.py1), r.px2 - r.px1, r.py2 - r.py1, fill=False, ec="#eb6834", lw=1.4, ls="--"))
        if r.matched == 1:
            ax.add_patch(plt.Rectangle((r.dx1, r.dy1), r.dx2 - r.dx1, r.dy2 - r.dy1, fill=False, ec="#2a78d6", lw=1.2))
            ax.annotate("", xy=((r.dx1 + r.dx2) / 2, (r.dy1 + r.dy2) / 2), xytext=((r.px1 + r.px2) / 2, (r.py1 + r.py2) / 2),
                        arrowprops=dict(arrowstyle="->", color="white", lw=1.0))
    xs = np.r_[rows.px1, rows.px2, rows.dx1.dropna(), rows.dx2.dropna()]
    ys = np.r_[rows.py1, rows.py2, rows.dy1.dropna(), rows.dy2.dropna()]
    mx, my = 0.15 * (xs.max() - xs.min()) + 20, 0.15 * (ys.max() - ys.min()) + 20
    ax.set_xlim(max(0, xs.min() - mx), min(img.shape[1], xs.max() + mx))
    ax.set_ylim(min(img.shape[0], ys.max() + my), max(0, ys.min() - my))
    ax.set_axis_off()
    ax.set_title(f"MOT17-{short(seq)}, frame {frame} (camera change {f.accel_pct:.2f}% of diagonal). "
                 f"Dashed: Kalman prediction; solid: matched detection", fontsize=8)
    save(fig, "fig_e3_case")
    print(f"case figure: {seq} frame {frame}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-seq", default=None, help="draw fig_e3_case on this sequence (needs the images)")
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    seqs = [s for s in SEQS if (E3 / "camera_motion" / f"{s}.csv").exists() and (E3 / "assoc" / f"{s}.csv.gz").exists()]
    parts = [load_seq(s) for s in seqs]
    frames = pd.concat([p[0] for p in parts], ignore_index=True)
    one = pd.concat([p[1] for p in parts], ignore_index=True)
    frames.to_csv(E3 / "e3_frames.csv", index=False, float_format="%.6g")

    # ---- 5.2 sanity + per-sequence table ----------------------------------------------------------
    rows = []
    for s in seqs:
        f, o = frames[frames.seq == s], one[one.seq == s]
        r_disp, p_disp = rho(f.disp_pct.values, f.iou.values)
        r_acc, p_acc = rho(f.accel_pct.values, f.iou.values)
        r_err, _ = rho(o.cam_change_rel.values, o.pred_err.values)
        rows.append({"seq": short(s), "group": group_of(s), "fit_ok_pct": 100 * f.ok.iloc[1:].mean(),
                     "motion_median": f.disp_pct.median(), "motion_p90": f.disp_pct.quantile(0.9),
                     "change_median": f.accel_pct.median(),
                     "pred_iou": o.loc[o.matched == 1, "iou"].mean(), "unmatched_pct": 100 * (o.matched == 0).mean(),
                     "pred_err": o.pred_err.median(),
                     "rho_motion_iou": r_disp, "p_motion_iou": p_disp, "rho_change_iou": r_acc, "p_change_iou": p_acc,
                     "rho_change_err": r_err})
    per_seq = pd.DataFrame(rows).sort_values(["group", "seq"], ascending=[False, True])
    per_seq.to_csv(E3 / "e3_per_seq.csv", index=False, float_format="%.4g")
    cols = ["seq", "group", "motion_median", "motion_p90", "change_median", "pred_iou", "unmatched_pct", "rho_motion_iou", "rho_change_iou"]
    head = ["Seq.", "Camera", "Motion med.", "Motion p90", "Change med.", "Pred. IoU", "Unmatched \\%", "$\\rho$(motion, IoU)", "$\\rho$(change, IoU)"]
    fmt = {"motion_median": "{:.2f}", "motion_p90": "{:.2f}", "change_median": "{:.2f}", "pred_iou": "{:.3f}",
           "unmatched_pct": "{:.1f}", "rho_motion_iou": "{:+.2f}", "rho_change_iou": "{:+.2f}"}
    lines = [r"\begin{tabular}{ll" + "r" * (len(cols) - 2) + "}", r"\toprule", " & ".join(head) + r" \\", r"\midrule"]
    for _, r in per_seq.iterrows():
        lines.append(" & ".join(fmt[c].format(r[c]) if c in fmt else str(r[c]) for c in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (E3 / "e3_per_seq.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- within-sequence comparison: frames above vs below the sequence's median camera motion -------
    strat = []
    for s in [s for s in seqs if group_of(s) == "moving"]:
        f = frames[(frames.seq == s) & frames.disp_pct.notna()]
        med = f.disp_pct.median()
        for half, sel in (("low motion", f.disp_pct <= med), ("high motion", f.disp_pct > med)):
            g = f[sel]
            strat.append({"seq": short(s), "half": half, "frames": len(g), "gt": g["gt"].sum(), "switch": g.switch.sum(),
                          "miss": g.miss.sum(), "iou": g.iou.mean(), "unmatched": g.unmatched.mean()})
    strat = pd.DataFrame(strat)
    if len(strat):
        tot = strat.groupby("half").agg(gt=("gt", "sum"), switch=("switch", "sum"), miss=("miss", "sum"),
                                        iou=("iou", "mean"), unmatched=("unmatched", "mean")).reset_index()
        tot["switch_per_1k"] = 1000 * tot["switch"] / tot["gt"]
        tot["miss_pct"] = 100 * tot["miss"] / tot["gt"]
        strat.to_csv(E3 / "e3_within_seq.csv", index=False, float_format="%.4g")
        tot.to_csv(E3 / "e3_within_seq_total.csv", index=False, float_format="%.4g")

    # ---- figures ------------------------------------------------------------------------------------
    # 5.2 timeline of camera motion
    fig, ax = plt.subplots(figsize=(6.8, 2.4))
    for s in seqs:
        f = frames[frames.seq == s]
        ax.plot(f.frame, f.disp_pct, SEQ_STYLE[s], color=GROUP_COLORS[group_of(s)], lw=0.9, label=short(s))
    ax.set_xlabel("frame (val half)")
    ax.set_ylabel("camera motion (% diag / frame)")
    ax.set_yscale("symlog", linthresh=0.05)
    ax.legend(ncol=7, frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    save(fig, "fig_e3_timeline")

    # 5.4 binned frame-level relations, per camera group
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
    ok = frames.disp_pct.notna()
    bins = np.unique(np.quantile(frames.loc[ok, "disp_pct"], np.linspace(0, 1, 9)))
    for g in GROUPS:
        f = frames[ok & (frames.group == g)]
        binned(axes[0], f.disp_pct.values, f.iou.values, bins, GROUP_COLORS[g], g)
        binned(axes[1], f.disp_pct.values, 100 * f.unmatched.values, bins, GROUP_COLORS[g], g)
        sw = 1000 * f.switch.values / f["gt"].replace(0, np.nan).values
        binned(axes[2], f.disp_pct.values, sw, bins, GROUP_COLORS[g], g)
    for a, t in zip(axes, ["pred. IoU (matched)", "unmatched predictions (%)", "ID switches / 1k GT"]):
        a.set_xscale("log")
        a.set_xlabel("camera motion (% diag / frame)")
        a.set_title(t, fontsize=8)
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, "fig_e3_binned")

    # 5.4 match-level: prediction error vs camera change at the box
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.4), sharey=True)
    m = one[(one.matched == 1) & one.pred_err.notna()]
    for a, col, xl in ((axes[0], "cam_flow_rel", "camera flow at box (box heights)"),
                       (axes[1], "cam_change_rel", "camera change at box (box heights)")):
        x = m[col].values
        okx = np.isfinite(x) & (x > 0)
        bins = np.quantile(x[okx], np.linspace(0, 1, 11)) if okx.any() else np.array([0, 1])
        for g in GROUPS:
            sel = okx & (m.group == g).values
            binned(a, x[sel], m.pred_err.values[sel], bins, GROUP_COLORS[g], g)
        a.set_xscale("log")
        a.set_xlabel(xl)
    axes[0].set_ylabel("pred. error (box heights)")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, "fig_e3_error")

    if args.case_seq:
        case_figure(args.case_seq, frames, one)

    # ---- console summary ----------------------------------------------------------------------------
    pd.set_option("display.width", 160)
    print(per_seq[cols + ["p_motion_iou", "p_change_iou", "rho_change_err"]].round(3).to_string(index=False))
    if len(strat):
        print("\nMoving sequences, frames split at each sequence's median camera motion:")
        print(tot[["half", "gt", "switch", "switch_per_1k", "miss_pct", "iou", "unmatched"]].round(3).to_string(index=False))
    r_all, p_all = rho(m.cam_change_rel.values, m.pred_err.values)
    n_ok = int((np.isfinite(m.cam_change_rel) & np.isfinite(m.pred_err)).sum())
    print(f"\nMatch level (all seqs): rho(camera change at box, pred error) = {r_all:+.3f} (p={p_all:.1e}, n={n_ok})")
    r_all, p_all = rho(m.cam_flow_rel.values, m.pred_err.values)
    print(f"Match level (all seqs): rho(camera flow at box,   pred error) = {r_all:+.3f} (p={p_all:.1e})")
    print(f"Saved to {E3}")


if __name__ == "__main__":
    main()
