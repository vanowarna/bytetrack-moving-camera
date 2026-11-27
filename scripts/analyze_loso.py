"""Robustness of the static/moving comparison: leave-one-sequence-out (LOSO) and per-sequence views.

    uv run python scripts/analyze_loso.py                 # official run (e1_default)
    uv run python scripts/analyze_loso.py --run e1_interp20

With three static and four moving sequences, one sequence can drive a group difference. This script
re-evaluates every group with one sequence left out (TrackEval on the cached tracks, no new tracking),
repeats the lost-track analysis of E3 in the same way, and also reports the unweighted mean over
sequences (each sequence counts once, instead of each ground-truth box).

Inputs : results/<run>/tracks/, results/e3/{camera_motion,assoc,events}/ (from run_camera_motion.py, run_e3_log.py)
Outputs: results/e2/loso_<run>.csv          one row per left-out sequence (+ 'none')
         results/e2/loso_pairs_<run>.csv    one static and one moving sequence left out (3 x 4 = 12 rows)
         results/e2/lost_per_seq.csv        lost-track recovery (9-16 frames) per sequence
         results/e2/figures/fig_loso.{pdf,png}
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import GROUP_COLORS, MOVING, RESULTS_DIR, SEQS, STATIC  # noqa: E402
from src.eval.trackeval_runner import evaluate  # noqa: E402

OUT = RESULTS_DIR / "e2"
FIG = OUT / "figures"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "savefig.bbox": "tight", "savefig.dpi": 200, "pdf.fonttype": 42})
KEYS = ("HOTA", "DetA", "AssA", "IDF1")


def short(seq):
    return seq.replace("MOT17-", "").replace("-FRCNN", "")


def group_metrics(run_dir, seqs, cache):
    """TrackEval combined over `seqs` (cached per subset)."""
    key = tuple(sorted(seqs))
    if key not in cache:
        name = "loso/" + "_".join(short(s) for s in key)
        cache[key] = evaluate(run_dir, list(key), name=name)["combined"]
    return cache[key]


def lost_rates(lost, seqs, lo=9, hi=16):
    """Share (%) of lost-track predictions re-matched after lo..hi lost frames, and their mean best IoU."""
    sel = lost[lost.seq.isin(seqs) & lost.since.between(lo, hi)]
    return 100 * sel.matched.mean(), sel.best_iou.mean(), len(sel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="e1_default")
    args = ap.parse_args()
    run_dir = RESULTS_DIR / args.run
    FIG.mkdir(parents=True, exist_ok=True)

    # lost-track log of E3 (official run only)
    lost = None
    if args.run == "e1_default":
        from analyze_e3 import load_seq  # noqa: E402
        lost = pd.concat([load_seq(s)[2] for s in SEQS], ignore_index=True)

    per_seq = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))["trackeval_per_seq"]
    cache = {}
    rows = []
    for out in [None] + SEQS:
        st = [s for s in STATIC if s != out]
        mv = [s for s in MOVING if s != out]
        gs, gm = group_metrics(run_dir, st, cache), group_metrics(run_dir, mv, cache)
        row = {"left_out": "none" if out is None else short(out),
               "group_left_out": "" if out is None else ("static" if out in STATIC else "moving")}
        for k in KEYS:
            row[f"{k}_static"], row[f"{k}_moving"] = gs[k], gm[k]
            row[f"{k}_gap"] = gs[k] - gm[k]
            # unweighted: every sequence counts once
            us = sum(per_seq[s][k] for s in st) / len(st)
            um = sum(per_seq[s][k] for s in mv) / len(mv)
            row[f"{k}_gap_seqmean"] = us - um
        if lost is not None:
            rs, bs, ns = lost_rates(lost, st)
            rm, bm, nm = lost_rates(lost, mv)
            row.update({"refound_static": rs, "refound_moving": rm, "best_iou_static": bs, "best_iou_moving": bm,
                        "n_lost_static": ns, "n_lost_moving": nm})
        rows.append(row)
        print(f"left out {row['left_out']:>4}: HOTA gap {row['HOTA_gap']:5.1f}  AssA gap {row['AssA_gap']:5.1f}  "
              f"DetA gap {row['DetA_gap']:5.1f}  (seq-mean HOTA gap {row['HOTA_gap_seqmean']:5.1f})"
              + (f"  re-found 9-16: {row['refound_static']:.1f}% vs {row['refound_moving']:.1f}%" if lost is not None else ""))
    loso = pd.DataFrame(rows)
    loso.to_csv(OUT / f"loso_{args.run}.csv", index=False, float_format="%.4f")

    pairs = []
    for s_out, m_out in itertools.product(STATIC, MOVING):
        st, mv = [s for s in STATIC if s != s_out], [s for s in MOVING if s != m_out]
        gs, gm = group_metrics(run_dir, st, cache), group_metrics(run_dir, mv, cache)
        r = {"static_left_out": short(s_out), "moving_left_out": short(m_out)}
        r.update({f"{k}_gap": gs[k] - gm[k] for k in KEYS})
        if lost is not None:
            r["refound_static"], r["refound_moving"] = lost_rates(lost, st)[0], lost_rates(lost, mv)[0]
        pairs.append(r)
    pairs = pd.DataFrame(pairs)
    pairs.to_csv(OUT / f"loso_pairs_{args.run}.csv", index=False, float_format="%.4f")

    if lost is not None:
        lps = []
        for s in SEQS:
            r, b, n = lost_rates(lost, [s])
            lps.append({"seq": short(s), "group": "static" if s in STATIC else "moving", "refound_9_16": r,
                        "best_iou_9_16": b, "n": n})
        pd.DataFrame(lps).to_csv(OUT / "lost_per_seq.csv", index=False, float_format="%.4f")

    # figure: gap with each sequence left out
    fig, axes = plt.subplots(1, 2 if lost is not None else 1, figsize=(7.0, 2.4), squeeze=False)
    ax = axes[0, 0]
    labels = ["all"] + [short(s) for s in SEQS]
    colors = ["#888888"] + [GROUP_COLORS["static" if s in STATIC else "moving"] for s in SEQS]
    x = range(len(labels))
    ax.bar([i - 0.2 for i in x], loso.HOTA_gap, 0.4, color=colors, label="HOTA gap")
    ax.bar([i + 0.2 for i in x], loso.AssA_gap, 0.4, color=colors, alpha=0.45, label="AssA gap")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(list(x), labels)
    ax.set_xlabel("sequence left out")
    ax.set_ylabel("static $-$ moving")
    ax.legend(frameon=False, fontsize=7)
    if lost is not None:
        ax = axes[0, 1]
        ax.plot(list(x), loso.refound_static, "o-", color=GROUP_COLORS["static"], label="static")
        ax.plot(list(x), loso.refound_moving, "o-", color=GROUP_COLORS["moving"], label="moving")
        ax.set_xticks(list(x), labels)
        ax.set_ylim(0, None)
        ax.set_xlabel("sequence left out")
        ax.set_ylabel("re-found after 9-16 frames (%)")
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig_loso.{ext}")
    plt.close(fig)

    lo = loso[loso.left_out != "none"]
    print(f"\nLOSO ({args.run}): HOTA gap {lo.HOTA_gap.min():.1f}..{lo.HOTA_gap.max():.1f}, "
          f"AssA gap {lo.AssA_gap.min():.1f}..{lo.AssA_gap.max():.1f}, DetA gap {lo.DetA_gap.min():.1f}..{lo.DetA_gap.max():.1f}")
    print(f"Pairs (one static + one moving out): HOTA gap {pairs.HOTA_gap.min():.1f}..{pairs.HOTA_gap.max():.1f}")
    if lost is not None:
        print(f"Re-found 9-16, LOSO: static {lo.refound_static.min():.1f}..{lo.refound_static.max():.1f}% vs "
              f"moving {lo.refound_moving.min():.1f}..{lo.refound_moving.max():.1f}%; pairs: static > moving in "
              f"{int((pairs.refound_static > pairs.refound_moving).sum())}/{len(pairs)}")


if __name__ == "__main__":
    main()
