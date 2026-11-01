"""
Compare the deployed model (v1, backend/) with the retrained model (v2, backend/model_v2/)
on exactly the same held-out windows from prepare_data_v2.py.

Both models get the same raw 60×6 windows and run through their own HARModel pipeline
(preprocess_window -> own scaler -> model), i.e. what the live app would do.

Usage (from backend/):  .venv/bin/python compare_models.py <windows.npz>
Writes model_v2/comparison.json and model_v2/comparison.md
"""

import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from model_service import HARModel, SessionSmoother

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = {"v1": BASE, "v2": os.path.join(BASE, "model_v2")}


def metrics(y_true, y_pred, classes):
    present = [c for c in classes if (y_true == c).any()]
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=classes, zero_division=0)
    _, _, f_present, _ = precision_recall_fscore_support(y_true, y_pred, labels=present, zero_division=0)
    return {
        "n": int(len(y_true)),
        "accuracy": round(float((y_true == y_pred).mean()), 4),
        "macro_f1_present_classes": round(float(f_present.mean()), 4),
        "per_class": {c: {"precision": round(float(p[i]), 3), "recall": round(float(r[i]), 3),
                          "f1": round(float(f[i]), 3), "support": int(s[i])} for i, c in enumerate(classes)},
        "confusion": {"labels": classes, "rows_true_cols_pred":
                      confusion_matrix(y_true, y_pred, labels=classes).tolist()},
    }


def main():
    d = np.load(sys.argv[1], allow_pickle=False)
    X, y, src, subj, split = d["X"].astype(np.float64), d["y"], d["source"], d["subject"], d["split"]
    groups = {
        "public_heldout_all": (split == "test") & (src != "phone"),
        "public_heldout_wisdm": (split == "test") & (src == "wisdm"),
        "public_heldout_heterogeneity": (split == "test") & (src == "het"),
        "public_heldout_uci": (split == "test") & (src == "uci"),
        "phone_heldout_stairs_session": (split == "test") & np.isin(subj, ["stairs1", "stairs2"]),
        "phone_heldout_walking2": (split == "test") & (subj == "walking2"),
        "phone_seen_in_training": (split == "train") & (src == "phone"),
    }
    results = {"groups": {}, "pipeline_phone": {}, "models": {}}
    for name, mdir in MODELS.items():
        har = HARModel(mdir)
        classes = har.classes
        probs = har.predict_probs(X[split != "val"])
        pred_all = np.full(len(X), "", dtype=object)
        pred_all[split != "val"] = np.array(classes)[probs.argmax(1)]
        P = np.zeros((len(X), len(classes))); P[split != "val"] = probs
        for g, m in groups.items():
            results["groups"].setdefault(g, {})[name] = metrics(y[m], pred_all[m].astype(str), classes)

        # Full dashboard pipeline (rules + per-recording smoothing) on the phone recordings.
        for rid in ["stairs1", "stairs2", "walking2", "still", "jogging", "walking1"]:
            idx = np.where((src == "phone") & (subj == rid))[0]
            sm = SessionSmoother()
            outs = [har.postprocess(P[i], X[i], sm)["activity"] for i in idx]
            results["pipeline_phone"].setdefault(rid, {"label": str(y[idx[0]]), "n": int(len(idx))})[name] = {
                "accuracy": round(float(np.mean(np.array(outs) == y[idx[0]])), 4),
                "predictions": {a: outs.count(a) for a in sorted(set(outs))}}

        w = X[:1]
        har.predict_probs(w)
        t = []
        for _ in range(100):
            t0 = time.perf_counter(); har.predict_probs(w); t.append((time.perf_counter() - t0) * 1000)
        results["models"][name] = {"dir": os.path.relpath(mdir, BASE) or ".", "parameters": har.param_count,
                                   "latency_ms_median_single_window": round(float(np.median(t)), 2),
                                   "latency_ms_p95_single_window": round(float(np.percentile(t, 95)), 2)}

    out = os.path.join(BASE, "model_v2")
    with open(os.path.join(out, "comparison.json"), "w") as f:
        json.dump(results, f, indent=2)

    lines = ["# v1 (deployed) vs v2 (retrained) on the same held-out windows", ""]
    for g, res in results["groups"].items():
        lines += [f"## {g}", "", f"n = {res['v1']['n']}", "",
                  "| | v1 | v2 |", "|---|---|---|",
                  f"| accuracy | {res['v1']['accuracy']} | {res['v2']['accuracy']} |",
                  f"| macro F1 (classes present) | {res['v1']['macro_f1_present_classes']} | {res['v2']['macro_f1_present_classes']} |", "",
                  "| class | support | v1 P / R / F1 | v2 P / R / F1 |", "|---|---|---|---|"]
        for c, a in res["v1"]["per_class"].items():
            b = res["v2"]["per_class"][c]
            lines.append(f"| {c} | {a['support']} | {a['precision']} / {a['recall']} / {a['f1']} | "
                         f"{b['precision']} / {b['recall']} / {b['f1']} |")
        for v in ("v1", "v2"):
            cm = res[v]["confusion"]
            lines += ["", f"{v} confusion (rows = true, cols = predicted)", "",
                      "| true \\ pred | " + " | ".join(cm["labels"]) + " |",
                      "|---" * (len(cm["labels"]) + 1) + "|"]
            lines += [f"| {cm['labels'][i]} | " + " | ".join(map(str, row)) + " |"
                      for i, row in enumerate(cm["rows_true_cols_pred"]) if sum(row)]
        lines.append("")
    lines += ["## Dashboard pipeline (rules + smoothing) on phone recordings", "",
              "| recording | label | n | v1 accuracy | v1 predictions | v2 accuracy | v2 predictions |",
              "|---|---|---|---|---|---|---|"]
    for rid, r in results["pipeline_phone"].items():
        lines.append(f"| {rid} | {r['label']} | {r['n']} | {r['v1']['accuracy']} | {r['v1']['predictions']} | "
                     f"{r['v2']['accuracy']} | {r['v2']['predictions']} |")
    lines += ["", "## Model size and latency", "", "| | parameters | median ms | p95 ms |", "|---|---|---|---|"]
    for v, m in results["models"].items():
        lines.append(f"| {v} | {m['parameters']} | {m['latency_ms_median_single_window']} | {m['latency_ms_p95_single_window']} |")
    with open(os.path.join(out, "comparison.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
