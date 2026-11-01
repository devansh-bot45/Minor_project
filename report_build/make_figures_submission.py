"""
Figure 8.1 of Minor_Project_Report_SUBMISSION_FINAL (axis labels 12 pt, other text 11 pt at print size), drawn from backend/model_v2/comparison.json
(the evaluation output of the deployed model; no numbers are typed in by hand).

Usage (from the repository root):  backend/.venv/bin/python report_build/make_figures_submission.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "report_submission", "figures")
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
fig, ax = plt.subplots(figsize=(5.8, 4.8), dpi=200)
ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
ax.set_yticks(range(len(labels)), labels)
ax.set_xlabel("Predicted class", fontsize=12)
ax.set_ylabel("True class", fontsize=12)
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, f"{pct[i, j]:.1f}", ha="center", va="center", fontsize=11, fontweight="bold",
                color="white" if pct[i, j] > 55 else "#1f2328")
ax.spines[:].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig8_1_confusion.png"))
plt.close(fig)

print("figure written to", OUT)
