"""
Figures 8.1 and 8.2 of Minor_Project_Report_FINAL, drawn from backend/model_v2/comparison.json
(the evaluation output of the deployed model; no numbers are typed in by hand).

Usage (from the repository root):  backend/.venv/bin/python report_build/make_figures_final.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "report_final", "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.family": "Times New Roman", "font.size": 11, "font.weight": "bold",
                     "axes.labelweight": "bold", "axes.spines.top": False, "axes.spines.right": False})

with open(os.path.join(ROOT, "backend", "model_v2", "comparison.json")) as f:
    res = json.load(f)["groups"]["public_heldout_all"]["v2"]

# Figure 8.1: row-normalised confusion matrix (percentage of each true class)
cm = res["confusion"]
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
        ax.text(j, i, f"{pct[i, j]:.1f}", ha="center", va="center", fontsize=10, fontweight="bold",
                color="white" if pct[i, j] > 55 else "#1f2328")
ax.spines[:].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig8_1_confusion.png"))
plt.close(fig)

# Figure 8.2: per-class precision, recall and F1-score
metrics = [("precision", "Precision", "#1f4e79"), ("recall", "Recall", "#5b9bd5"), ("f1", "F1-score", "#a6a6a6")]
x = np.arange(len(labels))
w = 0.26
fig, ax = plt.subplots(figsize=(6.8, 3.8), dpi=200)
for k, (key, name, color) in enumerate(metrics):
    vals = [res["per_class"][c][key] for c in labels]
    pos = x + (k - 1) * w
    ax.bar(pos, vals, w * 0.92, label=name, color=color)
    for p, v in zip(pos, vals):
        ax.text(p, v + 0.015, f"{v:.2f}", ha="center", fontsize=7.5, fontweight="bold")
ax.set_xticks(x, labels)
ax.set_ylim(0, 1.1)
ax.set_ylabel("Score")
ax.yaxis.grid(True, color="#e5e7eb")
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.12))
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig8_2_per_class_metrics.png"))
plt.close(fig)
print("figures written to", OUT)
