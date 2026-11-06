"""E4: summarise the one-at-a-time sensitivity runs (configs/e4/*.yaml + the official e1_default).

    uv run python scripts/analyze_e4.py

Output: results/e4/e4_sensitivity.csv, e4_summary.json, figures/fig_e4_sensitivity.{pdf,png}
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src.config import GROUP_COLORS, RESULTS_DIR, ROOT  # noqa: E402

OUT = RESULTS_DIR / "e4"
FIG = OUT / "figures"
DEFAULT_RUN = "e1_default"
PARAMS = [("track_thresh", "high-score threshold", 0.6), ("match_thresh", "matching threshold", 0.9),
          ("track_buffer", "track buffer (frames)", 30), ("kalman_noise_scale", "Kalman process-noise scale", 1.0)]
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e1e0d9", "grid.linewidth": 0.6, "savefig.bbox": "tight", "savefig.dpi": 200,
                     "pdf.fonttype": 42})


def row(run, param, value):
    m = json.loads((RESULTS_DIR / run / "metrics.json").read_text(encoding="utf-8"))
    te, mm = m["trackeval"], m["motmetrics_bytetrack_protocol"]["combined"]
    return {"run": run, "param": param, "value": value,
            "HOTA": te["all"]["HOTA"], "MOTA_mm": mm["mota"], "IDF1_mm": mm["idf1"], "IDs_mm": int(mm["num_switches"]),
            "HOTA_static": te["static"]["HOTA"], "HOTA_moving": te["moving"]["HOTA"],
            "AssA_static": te["static"]["AssA"], "AssA_moving": te["moving"]["AssA"],
            "DetA_static": te["static"]["DetA"], "DetA_moving": te["moving"]["DetA"],
            "IDSW_static": te["static"]["IDSW"], "IDSW_moving": te["moving"]["IDSW"]}


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in sorted((ROOT / "configs" / "e4").glob("*.yaml")):
        cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
        run = cfg["run_id"]
        if not (RESULTS_DIR / run / "metrics.json").exists():
            print(f"skip {run}: no metrics.json (run run_many first)")
            continue
        rows.append(row(run, cfg["e4"]["param"], cfg["e4"]["value"]))
    for param, _, default in PARAMS:  # the official run is the default point of every curve
        rows.append(row(DEFAULT_RUN, param, default))
    df = pd.DataFrame(rows)
    df["gap_HOTA"] = df.HOTA_static - df.HOTA_moving
    df["gap_AssA"] = df.AssA_static - df.AssA_moving
    df["gap_DetA"] = df.DetA_static - df.DetA_moving
    df = df.sort_values(["param", "value"])
    df.to_csv(OUT / "e4_sensitivity.csv", index=False, float_format="%.3f")

    # summary
    d0 = df[df.run == DEFAULT_RUN].iloc[0]
    summary = {"gap_default": float(d0.gap_HOTA), "gap_min": float(df.gap_HOTA.min()), "gap_max": float(df.gap_HOTA.max()),
               "assa_gt_deta_all": bool((df.gap_AssA > df.gap_DetA).all()), "n_settings": int(df.run.nunique()),
               "assa_gt_deta_count": int((df.drop_duplicates("run").gap_AssA > df.drop_duplicates("run").gap_DetA).sum()),
               "per_param": {}}
    for param, _, _ in PARAMS:
        s = df[df.param == param]
        best = s.loc[s.HOTA.idxmax()]
        summary["per_param"][param] = {
            "gap_min": float(s.gap_HOTA.min()), "gap_max": float(s.gap_HOTA.max()),
            "hota_min": float(s.HOTA.min()), "hota_max": float(s.HOTA.max()),
            "best_value": float(best.value), "best_hota": float(best.HOTA),
            "moving_min": float(s.HOTA_moving.min()), "moving_max": float(s.HOTA_moving.max()),
            "static_min": float(s.HOTA_static.min()), "static_max": float(s.HOTA_static.max())}
    (OUT / "e4_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # figure: HOTA static / moving vs each parameter
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.9), sharey=True)
    for ax, (param, label, default) in zip(axes, PARAMS):
        s = df[df.param == param]
        ax.plot(s.value, s.HOTA_static, "-o", ms=3, color=GROUP_COLORS["static"], label="static")
        ax.plot(s.value, s.HOTA_moving, "-D", ms=3, color=GROUP_COLORS["moving"], label="moving")
        ax.axvline(default, color="#888780", lw=0.8, ls="--")
        ax.set_xlabel(label)
        if param == "kalman_noise_scale":
            ax.set_xscale("log", base=2)
    axes[0].set_ylabel("HOTA")
    axes[0].legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig_e4_sensitivity.{ext}")
    plt.close(fig)

    pd.set_option("display.width", 160)
    cols = ["param", "value", "HOTA", "MOTA_mm", "IDF1_mm", "IDs_mm", "HOTA_static", "HOTA_moving", "gap_HOTA", "gap_AssA", "gap_DetA"]
    print(df[cols].round(2).to_string(index=False))
    print(f"\nGap (static - moving HOTA) over {summary['n_settings']} settings: {summary['gap_min']:.1f} .. {summary['gap_max']:.1f} "
          f"(official {summary['gap_default']:.1f}); AssA gap > DetA gap in {summary['assa_gt_deta_count']} of {summary['n_settings']} settings")
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
