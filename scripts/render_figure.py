#!/usr/bin/env python3
"""Render image.png from scripts/figure-data.json (curated results only).

Run: uv run --with matplotlib python scripts/render_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "scripts" / "figure-data.json").read_text())
OUT = ROOT / "image.png"

MONTHS = DATA["months"]
X = np.arange(len(MONTHS))
XLAB = [m[2:] if m.startswith("20") else m for m in MONTHS]  # "19-07" style
# Keep year marks readable: label every January plus first/last.
XTICKS = [i for i, m in enumerate(MONTHS) if m.endswith("-01") or i in (0, len(MONTHS) - 1)]
XLABELS = [MONTHS[i] for i in XTICKS]

COL_STATIC = "#ffd166"
COL_MONTHLY = "#4cc2ff"
COL_DRIFT = "#ff9e64"
COL_BASE = "#8b96ad"
COL_CUT = "#ff6b6b"

fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
fig.suptitle(
    "NYC taxi ETA v1.1: monthly MAE, paired policy bootstrap, "
    "drift lead/lag and interval coverage",
    fontsize=13,
)

# (a) monthly MAE -------------------------------------------------------
ax = axes[0, 0]
ax.plot(X, DATA["static_mae"], color=COL_STATIC, lw=1.8, label="static")
ax.plot(X, DATA["monthly_mae"], color=COL_MONTHLY, lw=1.8, label="monthly")
ax.plot(X, DATA["drift_mae"], color=COL_DRIFT, lw=1.8, label="drift")
ax.plot(X, DATA["baseline_mae"], color=COL_BASE, lw=1.0, alpha=0.8,
        label="zone-pair baseline")
ax.axhline(DATA["degraded_cut"], color=COL_CUT, ls="--", lw=1.0,
           label="degraded cut")
covid_i = MONTHS.index("2020-03")
ax.axvline(covid_i - 0.5, color="#c8d3e8", ls=":", lw=1.0)
ax.text(covid_i - 0.45, ax.get_ylim()[1] if False else max(DATA["static_mae"]) * 1.02,
        "COVID", fontsize=8, color="#c8d3e8")
rt = [MONTHS.index(m) for m in DATA["retrain_months"]]
ax.plot(rt, [min(DATA["monthly_mae"]) - 0.12] * len(rt), "|",
        color=COL_DRIFT, ms=8, mew=1.6, label="drift retrain")
ax.set_xticks(xticks := XTICKS)
ax.set_xticklabels(XLABELS, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("MAE (minutes)")
ax.set_title("Monthly MAE by retraining policy", fontsize=11)
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.15)

# (b) forest plot --------------------------------------------------------
ax = axes[0, 1]
order = ["monthly-static", "drift-static", "drift-monthly"]
labels = ["monthly - static", "drift - static", "drift - monthly"]
ys = np.arange(len(order))[::-1]
for y, key in zip(ys, order, strict=True):
    d = DATA["bootstrap_diffs"][key]
    ax.errorbar(d["point"], y, xerr=[[d["point"] - d["ci_low"]], [d["ci_high"] - d["point"]]],
                fmt="o", color=COL_MONTHLY, ecolor=COL_MONTHLY, capsize=4, ms=6)
    ax.text(d["ci_high"] + 0.008, y, f"{d['point']:+.3f}", va="center", fontsize=8)
ax.axvline(0, color="#c8d3e8", lw=0.8)
ax.set_yticks(ys)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("Difference in mean monthly MAE (minutes), 95% CI")
ax.set_title("Paired day-cluster bootstrap (1000, seed 0)", fontsize=11)
ax.grid(alpha=0.15, axis="x")

# (c) spearman by lag ------------------------------------------------------
ax = axes[1, 0]
lags = ["-1", "0", "1"]  # JSON keys; display labels add the "+" sign
signals = ["drift_share", "pred_wasserstein", "volume_ratio"]
sig_labels = {"drift_share": "drift share", "pred_wasserstein": "pred. Wasserstein",
              "volume_ratio": "volume ratio"}
colors = {"drift_share": "#7ee787", "pred_wasserstein": "#79c0ff",
          "volume_ratio": "#f778ba"}
w = 0.26
for j, sig in enumerate(signals):
    vals = [DATA["spearman_lagged"][sig][lag] for lag in lags]
    ax.bar(np.array(lags_pos := np.arange(3)) + (j - 1) * w, vals, w,
           color=colors[sig], label=sig_labels[sig])
ax.axhline(0, color="#c8d3e8", lw=0.8)
ax.set_xticks(np.arange(3))
ax.set_xticklabels(["lag -1", "lag 0", "lag +1"], fontsize=9)
ax.set_ylim(-1, 1)
ax.set_ylabel("Spearman vs static MAE ratio")
ax.set_title("Signal lead/lag (n = 30 months, 29 at lags of 1, no CIs)",
             fontsize=11)
ax.legend(fontsize=8)
ax.grid(alpha=0.15, axis="y")

# (d) coverage --------------------------------------------------------------
ax = axes[1, 1]
ax.plot(X, DATA["coverage"], color=COL_MONTHLY, lw=1.6)
ax.axhline(0.80, color=COL_CUT, ls="--", lw=1.0, label="0.80 target")
ax.axhline(0.75, color=COL_BASE, ls=":", lw=0.8, label="0.75")
ax.set_xticks(XTICKS)
ax.set_xticklabels(XLABELS, rotation=45, ha="right", fontsize=8)
ax.set_ylim(0, 1)
ax.set_ylabel("80% interval coverage")
ax.set_title("Static model interval coverage", fontsize=11)
ax.legend(fontsize=8, loc="lower left")
ax.grid(alpha=0.15)

plt.tight_layout(rect=(0, 0, 1, 0.95))
plt.savefig(OUT, dpi=160)
print(f"wrote {OUT}")
