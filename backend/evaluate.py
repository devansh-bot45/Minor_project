"""
Evaluate the deployed model on the labelled phone recordings in the repository root.

Run from the backend folder:  python evaluate.py [model_dir]   (default: the original model in backend/)
Writes: <model_dir>/evaluation_report.json (served by GET /api/model) and evaluation_report.md

IMPORTANT: most of these recordings were also used for training (see prepare_data.py),
so their scores are optimistic. Only recordings marked used_in_training=False are
held out. The public datasets (WISDM / UCI HAR / Heterogeneity) and X_all.npy are not
in the repository, so the original test split cannot be reproduced here.
"""

import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
from scipy.signal import sosfilt
from sklearn.metrics import accuracy_score, confusion_matrix

from preprocessing import FRAME_SIZE, NUM_CHANNELS, parse_sensor_csv, sos_gravity, sos_noise, windows_from_segments
from model_service import HARModel, SessionSmoother
from recordings import available, model_metadata, path_for

BASE = os.path.dirname(os.path.abspath(__file__))


def legacy_causal_features(windows):
    """The ORIGINAL backend pipeline: causal sosfilt with state carried across
    consecutive non-overlapping windows (as the old frontend sent them)."""
    zn = [np.zeros((sos_noise.shape[0], 2)) for _ in range(6)]
    zg = [np.zeros((sos_gravity.shape[0], 2)) for _ in range(3)]
    out = []
    for w in windows:
        f = np.zeros_like(w)
        for i in range(6):
            f[:, i], zn[i] = sosfilt(sos_noise, w[:, i], zi=zn[i])
        g = np.zeros((len(w), 3))
        for i in range(3):
            g[:, i], zg[i] = sosfilt(sos_gravity, f[:, i], zi=zg[i])
        body = f[:, :3] - g
        out.append(np.hstack((body, f[:, 3:], np.linalg.norm(body, axis=1, keepdims=True),
                              np.linalg.norm(f[:, 3:], axis=1, keepdims=True))))
    return np.stack(out)


def summarize(y_true, y_pred):
    return {"windows": len(y_true), "accuracy": round(float(accuracy_score(y_true, y_pred)), 4)}


def main(model_dir=BASE):
    har = HARModel(model_dir)
    meta = model_metadata(model_dir)
    classes = har.classes
    labels_all = classes + ["Uncertain"]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "classes": classes,
        "notes": [
            "Windows: 60 samples at 20 Hz (resampled from the recording timestamps), 50% overlap; gaps > 1 s in a recording are skipped.",
            "'model' = pure model argmax; 'pipeline' = model + rules + per-session smoothing (what the dashboard shows).",
            "Recordings with used_in_training=true were part of the training data, so their scores are optimistic.",
        ] + meta.get("notes", [
            "The public-dataset test split from training cannot be reproduced (datasets and X_all.npy are not in the repo).",
        ]),
        "recordings": [],
    }
    groups = {True: {"t": [], "m": [], "p": []}, False: {"t": [], "m": [], "p": []}}

    for rec in available(model_dir):
        path = path_for(rec)
        if not os.path.exists(path):
            print(f"  [skip] {rec['file']} not found")
            continue
        with open(path, "rb") as fh:
            segments, info = parse_sensor_csv(fh.read())
        _, windows = windows_from_segments(segments)
        probs = har.predict_probs(windows)
        model_pred = [classes[i] for i in probs.argmax(axis=1)]
        smoother = SessionSmoother()
        pipe_pred = [har.postprocess(p, w, smoother)["activity"] for p, w in zip(probs, windows)]

        # Original backend pipeline, for comparison (pure model output only)
        legacy_truth_windows = windows_from_segments(segments, step=FRAME_SIZE)[1]
        legacy = legacy_causal_features(legacy_truth_windows)
        n = len(legacy)
        legacy = har.scaler.transform(legacy.reshape(-1, NUM_CHANNELS)).reshape(n, FRAME_SIZE, NUM_CHANNELS)
        legacy_pred = [classes[i] for i in har.model.predict(legacy, verbose=0).argmax(axis=1)]

        entry = {
            "file": rec["file"], "label": rec["label"], "used_in_training": rec["used_in_training"],
            "duration_sec": info["duration_sec"], "windows": len(windows),
            "model_distribution": {c: model_pred.count(c) for c in labels_all if model_pred.count(c)},
            "pipeline_distribution": {c: pipe_pred.count(c) for c in labels_all if pipe_pred.count(c)},
        }
        if rec["label"]:
            truth = [rec["label"]] * len(windows)
            entry["model_accuracy"] = summarize(truth, model_pred)["accuracy"]
            entry["pipeline_accuracy"] = summarize(truth, pipe_pred)["accuracy"]
            entry["legacy_model_accuracy"] = summarize([rec["label"]] * n, legacy_pred)["accuracy"]
            g = groups[rec["used_in_training"]]
            g["t"] += truth; g["m"] += model_pred; g["p"] += pipe_pred
        report["recordings"].append(entry)
        print(f"  {rec['file']:40s} {str(rec['label']):8s} windows={len(windows):4d} "
              f"model={entry.get('model_accuracy', '-')} pipeline={entry.get('pipeline_accuracy', '-')} "
              f"legacy={entry.get('legacy_model_accuracy', '-')}")

    for key, name in ((True, "seen_in_training"), (False, "held_out")):
        g = groups[key]
        if not g["t"]:
            continue
        present = sorted(set(g["t"]))
        report[name] = {
            "windows": len(g["t"]),
            "model_accuracy": summarize(g["t"], g["m"])["accuracy"],
            "pipeline_accuracy": summarize(g["t"], g["p"])["accuracy"],
            "confusion_matrix": {
                "rows_true": present, "cols_pred": labels_all,
                "pipeline": confusion_matrix(g["t"], g["p"], labels=labels_all)[
                    [labels_all.index(c) for c in present]].tolist(),
            },
        }

    with open(os.path.join(model_dir, "evaluation_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    write_markdown(report, model_dir)
    print("\n  Saved evaluation_report.json and evaluation_report.md")


def write_markdown(r, model_dir):
    lines = ["# Model evaluation on repository recordings", "", f"Generated: {r['generated_at']}", ""]
    lines += [f"- {n}" for n in r["notes"]] + [""]
    lines += ["| Recording | True label | In training? | Windows | Model acc. | Pipeline acc. | Old pipeline acc. |",
              "|---|---|---|---|---|---|---|"]
    fmt = lambda v: "-" if v is None else f"{v * 100:.1f}%"
    for e in r["recordings"]:
        lines.append(f"| {e['file']} | {e['label'] or 'unlabelled'} | "
                     f"{'-' if e['used_in_training'] is None else ('yes' if e['used_in_training'] else 'no')} | "
                     f"{e['windows']} | {fmt(e.get('model_accuracy'))} | {fmt(e.get('pipeline_accuracy'))} | "
                     f"{fmt(e.get('legacy_model_accuracy'))} |")
    for name in ("seen_in_training", "held_out"):
        if name in r:
            s = r[name]
            lines += ["", f"**{name.replace('_', ' ').title()}**: {s['windows']} windows, "
                      f"model {s['model_accuracy'] * 100:.1f}%, pipeline {s['pipeline_accuracy'] * 100:.1f}%"]
    lines += ["", "Prediction distribution per recording (pipeline):", ""]
    for e in r["recordings"]:
        lines.append(f"- {e['file']}: {e['pipeline_distribution']}")
    with open(os.path.join(model_dir, "evaluation_report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main(os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else BASE))
