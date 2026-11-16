#!/usr/bin/env python3
"""Plot detector v1 vs v2 on the same labeled images (eval_v1/ and eval_v2/)."""
import json, sys, pathlib
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

a, b = (json.loads(pathlib.Path(p, "metrics.json").read_text()) for p in sys.argv[1:3])
out = sys.argv[3] if len(sys.argv) > 3 else "pd5_v1_vs_v2.png"
ink, ink2, grid, bg, c1, c2 = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 11, "text.color": ink, "axes.labelcolor": ink2,
                     "xtick.color": ink2, "ytick.color": ink2, "axes.edgecolor": grid})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6), facecolor=bg, gridspec_kw={"width_ratios": [1, 1.4]})
for ax in (ax1, ax2):
    ax.set_facecolor(bg); ax.grid(axis="y", color=grid); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
keys = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]
x = np.arange(len(keys))
for off, m, col, lab in [(-0.19, a, c1, "v1"), (0.19, b, c2, "v2")]:
    vals = [m["overall"][k] * 100 for k, _ in keys]
    bars = ax1.bar(x + off, vals, 0.36, color=col, label=lab)
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6, f"{bar.get_height():.1f}",
                 ha="center", fontsize=9, color=ink2)
ax1.set_xticks(x, [l for _, l in keys]); ax1.set_ylim(80, 102); ax1.set_ylabel("%")
ax1.set_title(f"Overall ({a['overall']['n']} images)", loc="left", color=ink); ax1.legend(frameon=False, ncol=2, loc="lower right", bbox_to_anchor=(1, 1.0))
alts = list(a["by"]["altitude_m"].keys()); x2 = np.arange(len(alts))
for off, m, col, lab in [(-0.19, a, c1, "v1 false positives"), (0.19, b, c2, "v2 false positives")]:
    vals = [m["by"]["altitude_m"][k]["FP"] for k in alts]
    bars = ax2.bar(x2 + off, vals, 0.36, color=col, label=lab)
    for bar in bars:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f"{int(bar.get_height())}",
                 ha="center", fontsize=9, color=ink2)
ax2.set_xticks(x2, [f"{k} m" for k in alts]); ax2.set_ylabel("False positives (count)")
ax2.set_title("False positives by altitude", loc="left", color=ink); ax2.legend(frameon=False, ncol=2, loc="lower right", bbox_to_anchor=(1, 1.0))
fig.suptitle(f"Detector v1 -> v2: false positives {a['overall']['FP']} -> {b['overall']['FP']}, "
             f"precision {a['overall']['precision']:.1%} -> {b['overall']['precision']:.1%}",
             x=0.01, ha="left", fontweight="bold", color=ink)
fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(out, dpi=150, facecolor=bg)
print("wrote", out)
