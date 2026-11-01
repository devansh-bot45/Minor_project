"""
Figures 8.1 and 8.2 of the report, drawn from backend/model_v2/comparison.json (no numbers are typed in by hand).

Usage (from the repository root):  backend/.venv/bin/python report_build/make_figures.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "report_build", "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.family": "Times New Roman", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

with open(os.path.join(ROOT, "backend", "model_v2", "comparison.json")) as f:
    groups = json.load(f)["groups"]["public_heldout_all"]

# Figure 8.1: row-normalised confusion matrix of model v2 (percentage of each true class)
cm = groups["v2"]["confusion"]
labels = cm["labels"]
counts = np.array(cm["rows_true_cols_pred"], dtype=float)
pct = counts / counts.sum(axis=1, keepdims=True) * 100
fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=200)
ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
ax.set_yticks(range(len(labels)), labels)
ax.set_xlabel("Predicted class")
ax.set_ylabel("True class")
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, f"{pct[i, j]:.1f}", ha="center", va="center", fontsize=10,
                color="white" if pct[i, j] > 55 else "#1f2328")
ax.spines[:].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig8_1_confusion_v2.png"))
plt.close(fig)

# Figure 8.2: per-class F1-score of the original model and model v2
f1 = {v: [groups[v]["per_class"][c]["f1"] for c in labels] for v in ("v1", "v2")}
x = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(6.6, 3.6), dpi=200)
ax.bar(x - 0.19, f1["v1"], 0.36, label="Original model", color="#9aa4b2")
ax.bar(x + 0.19, f1["v2"], 0.36, label="Model v2", color="#2b6cb0")
for k in range(len(labels)):
    ax.text(x[k] - 0.19, f1["v1"][k] + 0.015, f"{f1['v1'][k]:.2f}", ha="center", fontsize=8.5)
    ax.text(x[k] + 0.19, f1["v2"][k] + 0.015, f"{f1['v2'][k]:.2f}", ha="center", fontsize=8.5)
ax.set_xticks(x, labels)
ax.set_ylim(0, 1.05)
ax.set_ylabel("F1-score")
ax.yaxis.grid(True, color="#e5e7eb")
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig8_2_f1_comparison.png"))
plt.close(fig)
print("figures written to", OUT)
