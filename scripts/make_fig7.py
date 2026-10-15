"""Regenerate Fig. 7 (controller study) from the files in results/.
Run from the repository root:  python scripts/make_fig7.py
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results/"
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

# (a) training return, 1000-episode moving average
for pv, c in ((4, "C0"), (10, "C1")):
    ret = np.load(f"{R}learning_curve_pv{pv}.npy"); k = 1000
    ax[0].plot(np.convolve(ret, np.ones(k) / k, "valid"), c, label=f"PV {pv} MW")
ax[0].axvline(40000, ls="--", c="gray", lw=1)
ax[0].text(40500, ax[0].get_ylim()[0] + 0.1, "exploration floor", fontsize=8, color="gray")
ax[0].set_title("(a) Training return"); ax[0].set_xlabel("Episode")
ax[0].set_ylabel("Return per 2-day episode"); ax[0].legend(loc="upper left")

# (b) emissions, (c) annual return
names = {"static": "Static", "rule": "Rule-based", "rl": "Q-learning"}
cols = {"static": "C7", "rule": "C3", "rl": "C0"}
marks = {"static": "s", "rule": "o", "rl": "^"}
res = {pv: json.load(open(f"{R}results_pv{pv}.json")) for pv in (4, 10)}
for i, (m, t, yl) in enumerate((("em", "(b) Annual emissions", "t CO$_2$/yr"),
                                ("R", "(c) Annual return", "Return (higher is better)"))):
    for j, k in enumerate(("static", "rule", "rl")):
        for p, pv in enumerate((4, 10)):
            ax[i + 1].errorbar(p + (j - 1) * 0.2, res[pv][k][m][0], yerr=res[pv][k][m][1],
                               fmt=marks[k], color=cols[k], capsize=4, ms=8,
                               label=names[k] if p == 0 else None)
    ax[i + 1].set_xticks([0, 1]); ax[i + 1].set_xticklabels(["PV 4 MW", "PV 10 MW"])
    ax[i + 1].set_xlim(-0.5, 1.5); ax[i + 1].grid(axis="y", alpha=0.3)
    ax[i + 1].set_title(t); ax[i + 1].set_ylabel(yl)
    ax[i + 1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3,
                     fontsize=9, frameon=False)

plt.tight_layout()
plt.savefig(f"{R}controller_training.png", dpi=300, bbox_inches="tight")
