"""Training-style custom windows: counts, effective sample rate, and model output."""
import os, sys, warnings, io, contextlib
warnings.filterwarnings("ignore")
R = sys.argv[1]; sys.path.insert(0, os.path.join(R, "backend"))
import numpy as np, pandas as pd
import importlib.util
spec = importlib.util.spec_from_file_location("prep", os.path.join(R, "prepare_data.py"))
prep = importlib.util.module_from_spec(spec); spec.loader.exec_module(prep)
from model_service import HARModel
har = HARModel(os.path.join(R, "backend")); C = har.classes
os.chdir(R)
ORIGINAL_TRAINING_MAP = {  # as in the ORIGINAL prepare_data.py that produced the current model
    "still data.csv": "Still", "jogging.csv": "Jogging",
    "sensor_recording_20260424_131838.csv": "Stairs", "sensor_recording_20260424_184254.csv": "Stairs",
    "sensor_recording_20260426_162439.csv": "Walking", "sensor_recording_20260427_174501.csv": "Walking",
}
AUG = {"Stairs": 3, "Still": 30, "Jogging": 25, "Walking": 25}
HELD = {"sensor_recording_20260426_145530.csv": "Stairs", "sensor_recording_20260426_145547.csv": "Stairs"}
print(f"{'file':40s} {'label':8s} raw_win  +aug  total  window_span_s  model-argmax distribution (training-style windows)")
tot = {}
for f, lab in list(ORIGINAL_TRAINING_MAP.items()) + list(HELD.items()):
    with contextlib.redirect_stdout(io.StringIO()):
        frames, _ = prep.extract_custom_csv(f, lab)
    # effective duration of a 60-row window, from the same merge the training used
    df = pd.read_csv(f); a = df[df['Sensor Type'] == 1].groupby('Timestamp').mean().reset_index()
    span = np.median([(a.Timestamp.values[i+59]-a.Timestamp.values[i])/1000 for i in range(0, len(a)-60, 30)])
    n = len(frames); aug = AUG.get(lab, 0) * n if f in ORIGINAL_TRAINING_MAP else 0
    X = np.stack(frames); X = har.scaler.transform(X.reshape(-1, 8)).reshape(X.shape)
    pred = [C[i] for i in har.model.predict(X, verbose=0).argmax(1)]
    dist = {c: pred.count(c) for c in C if pred.count(c)}
    if f in ORIGINAL_TRAINING_MAP: tot[lab] = tot.get(lab, 0) + n + aug
    print(f"{f:40s} {lab:8s} {n:6d} {aug:5d} {n+aug:6d}  {span:8.1f}      {dist}")
print("\nCustom windows per label fed into training (before global balancing):", tot)
