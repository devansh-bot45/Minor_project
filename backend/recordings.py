"""
The labelled phone recordings shipped in the repository root.
Used by evaluate.py and by the dashboard's "replay a recording" demo.
"""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# label = activity recorded; used_in_training = was this file used to train the CURRENT har_model.keras
RECORDINGS = [
    {"id": "still",    "file": "still data.csv",                       "label": "Still",   "used_in_training": True},
    {"id": "jogging",  "file": "jogging.csv",                          "label": "Jogging", "used_in_training": True},
    {"id": "walking1", "file": "sensor_recording_20260426_162439.csv", "label": "Walking", "used_in_training": True},
    {"id": "walking2", "file": "sensor_recording_20260427_174501.csv", "label": "Walking", "used_in_training": True},
    {"id": "stairs1",  "file": "sensor_recording_20260426_145530.csv", "label": "Stairs",  "used_in_training": False},
    {"id": "stairs2",  "file": "sensor_recording_20260426_145547.csv", "label": "Stairs",  "used_in_training": False},
]
# Recorded but with no reliable label (named "eating", previously tagged Stairs): predictions only.
UNLABELLED = [{"id": "eating", "file": "eating.csv", "label": None, "used_in_training": None}]


# The flags above describe the ORIGINAL model (backend/har_model.keras). A model trained later
# lists the recordings it was trained on in <model_dir>/trained_on_recordings.json.
def model_metadata(model_dir):
    path = os.path.join(model_dir, "trained_on_recordings.json") if model_dir else ""
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def available(model_dir=None):
    flags = model_metadata(model_dir).get("used_in_training", {})
    out = []
    for r in RECORDINGS + UNLABELLED:
        if os.path.exists(os.path.join(ROOT, r["file"])):
            r = dict(r)
            if r["label"] is not None and r["id"] in flags:
                r["used_in_training"] = flags[r["id"]]
            out.append(r)
    return out


def path_for(rec):
    return os.path.join(ROOT, rec["file"])
