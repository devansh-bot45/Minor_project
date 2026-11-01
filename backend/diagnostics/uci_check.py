"""UCI HAR: (a) exactly as prepare_data.extract_uci_har built training windows; (b) converted to phone units at 20 Hz."""
import os, sys, warnings
warnings.filterwarnings("ignore")
R, U = sys.argv[1], sys.argv[2]; sys.path.insert(0, os.path.join(R, "backend"))
import numpy as np
from scipy.ndimage import zoom
from sklearn.metrics import classification_report
from preprocessing import preprocess_window, parse_sensor_csv, windows_from_segments
from model_service import HARModel
from recordings import RECORDINGS, path_for
har = HARModel(os.path.join(R, "backend")); C = har.classes
MAP = {1: "Walking", 2: "Stairs", 3: "Stairs", 4: "Still", 5: "Still", 6: "Still"}
def load(split, names):
    return np.stack([np.loadtxt(f"{U}/{split}/Inertial Signals/{n}_{split}.txt") for n in names], axis=-1)
def run(X):  # X: N×60×8 features
    Xs = har.scaler.transform(X.reshape(-1, 8)).reshape(X.shape)
    return np.array([C[i] for i in har.model.predict(Xs, verbose=0, batch_size=256).argmax(1)])
feat_stats = {}
for split in ("train", "test"):
    y = np.loadtxt(f"{U}/{split}/y_{split}.txt").astype(int); subj = np.loadtxt(f"{U}/{split}/subject_{split}.txt").astype(int)
    lab = np.array([MAP[v] for v in y])
    body = load(split, ["body_acc_x", "body_acc_y", "body_acc_z", "body_gyro_x", "body_gyro_y", "body_gyro_z"])
    # (a) training-style
    Xa = np.stack([preprocess_window(zoom(w, (60/128, 1))) for w in body])
    pa = run(Xa)
    print(f"\n### UCI {split}: AS TRAINED (body_acc in g, 128@50Hz zoomed to 60) — n={len(lab)}")
    print(classification_report(lab, pa, labels=["Stairs", "Still", "Walking"], zero_division=0, digits=3))
    for c in ("Stairs", "Walking", "Still"):
        m = lab == c; print(f"  true {c:8s} -> " + str({k: int((pa[m] == k).sum()) for k in C if (pa[m] == k).sum()}))
    feat_stats[f"UCI as trained ({split})"] = {c: Xa[lab == c][:, :, 6].mean() for c in ("Stairs", "Walking", "Still")}
    # (b) phone units: total_acc (g, incl gravity) * 9.80665, body_gyro rad/s; rebuild continuous streams, resample to 20 Hz
    tot = load(split, ["total_acc_x", "total_acc_y", "total_acc_z"]) * 9.80665
    raw = np.concatenate([tot, body[:, :, 3:]], axis=-1)
    runs, cur, key = [], [], None
    for i in range(len(raw)):
        k = (subj[i], y[i])
        if k != key and cur: runs.append((key, np.concatenate(cur))); cur = []
        key = k; cur.append(raw[i][:64])
    runs.append((key, np.concatenate(cur)))
    Xb, lb = [], []
    for (s, yy), sig in runs:
        t50 = np.arange(len(sig)) / 50.0; t20 = np.arange(0, t50[-1], 1 / 20.0)
        sig20 = np.column_stack([np.interp(t20, t50, sig[:, c]) for c in range(6)])
        for st in range(0, len(sig20) - 60 + 1, 30):
            Xb.append(preprocess_window(sig20[st:st+60])); lb.append(MAP[yy])
    Xb, lb = np.stack(Xb), np.array(lb); pb = run(Xb)
    print(f"### UCI {split}: PHONE UNITS (m/s² incl. gravity, true 20 Hz, 3 s windows) — n={len(lb)}")
    print(classification_report(lb, pb, labels=["Stairs", "Still", "Walking"], zero_division=0, digits=3))
    for c in ("Stairs", "Walking"):
        m = lb == c; print(f"  true {c:8s} -> " + str({k: int((pb[m] == k).sum()) for k in C if (pb[m] == k).sum()}))
    feat_stats[f"UCI phone units ({split})"] = {c: Xb[lb == c][:, :, 6].mean() for c in ("Stairs", "Walking", "Still")}
    print(f"  UCI {split} Stairs windows (as trained): {int((lab == 'Stairs').sum())}")
# phone recordings: mean |body accel| (feature 6, m/s²)
ph = {}
for rec in RECORDINGS:
    segs, _ = parse_sensor_csv(open(path_for(rec), "rb").read()); _, w = windows_from_segments(segs)
    ph.setdefault(rec["label"], []).append(np.stack([preprocess_window(x) for x in w])[:, :, 6].mean())
print("\nMean |body acceleration| per window (input channel 7, before scaling):")
for k, v in feat_stats.items(): print(f"  {k:28s} " + "  ".join(f"{c}={x:.3f}" for c, x in v.items()))
print("  phone recordings (m/s²)     " + "  ".join(f"{c}={np.mean(v):.3f}" for c, v in ph.items()))
print("  scaler mean/std for this channel:", har.scaler.mean_[6].round(3), har.scaler.scale_[6].round(3))
