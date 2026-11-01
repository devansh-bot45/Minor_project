"""
Stairs investigation (read-only; never modifies model files).

Usage (from backend/):  .venv/bin/python diagnostics/stairs_investigation.py .. "<path>/UCI HAR Dataset"
The UCI path is optional; without it the UCI sections are skipped.
"""
import os, re, sys, pickle, warnings
warnings.filterwarnings("ignore")
R = sys.argv[1]; U = sys.argv[2] if len(sys.argv) > 2 else None
sys.path.insert(0, os.path.join(R, "backend"))
import numpy as np
import pandas as pd
from preprocessing import FS, preprocess_window, parse_sensor_csv, windows_from_segments
from model_service import HARModel, SessionSmoother
from recordings import RECORDINGS, path_for

B = os.path.join(R, "backend")
har = HARModel(B); C = har.classes


def section(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# ── 1. Label mapping consistency ──────────────────────────────────────────────
section("1. Label mapping: training -> artifacts -> model -> frontend")
src = open(os.path.join(B, "train_model.py")).read()
train_names = sorted(eval(re.search(r"sorted\((\[.*?\])\)", src).group(1)))
names = pickle.load(open(os.path.join(B, "activity_names.pkl"), "rb"))
enc = pickle.load(open(os.path.join(B, "label_encoder.pkl"), "rb"))
front = re.findall(r"^\s*'?([A-Za-z ]+?)'?:\s*'#", open(os.path.join(R, "frontend/src/constants.js")).read(), re.M)
print("train_model.py sorted names :", train_names)
print("activity_names.pkl          :", [names[i] for i in range(len(names))])
print("label_encoder.classes_      :", list(enc.classes_))
print("model output units          :", har.model.output_shape[-1])
print("frontend ACTIVITY_COLORS    :", front)
print("Stairs index everywhere     :", train_names.index("Stairs"), list(enc.classes_).index("Stairs"), C.index("Stairs"))
print("all model classes have a frontend colour:", set(C) <= set(front))

# ── 2. Phone recordings: time overlap ─────────────────────────────────────────
section("2. Phone recordings: time span, rates, overlap")
spans = {}
for r in RECORDINGS:
    d = pd.read_csv(path_for(r)); t = d["Timestamp"]
    n_a, n_g = int((d["Sensor Type"] == 1).sum()), int((d["Sensor Type"] == 4).sum())
    dur = (t.max() - t.min()) / 1000
    spans[r["id"]] = (t.min(), t.max())
    print(f"{r['id']:9s} {r['label']:8s} train={str(r['used_in_training']):5s} "
          f"{pd.to_datetime(t.min(), unit='ms')} +{dur:5.1f}s  accel {n_a/dur:5.1f} Hz  gyro {n_g/dur:5.1f} Hz")
a, b = spans["stairs1"], spans["stairs2"]
ov = max(0, min(a[1], b[1]) - max(a[0], b[0])) / 1000
print(f"stairs1 / stairs2 overlap in time: {ov:.1f} s of {(a[1]-a[0])/1000:.1f} s and {(b[1]-b[0])/1000:.1f} s")

# ── 3. Stairs counts at each stage of the ORIGINAL training pipeline ─────────
section("3. Stairs windows at each stage of the original (deployed) pipeline")
sys.path.insert(0, os.path.join(B, "diagnostics"))
orig_stairs_files = ["sensor_recording_20260424_131838.csv", "sensor_recording_20260424_184254.csv"]
raw = 0
for f in orig_stairs_files:
    d = pd.read_csv(os.path.join(R, f))
    acc = d[d["Sensor Type"] == 1].groupby("Timestamp")[["Value1", "Value2", "Value3"]].mean().reset_index()
    gyr = d[d["Sensor Type"] == 4].groupby("Timestamp")[["Value1", "Value2", "Value3"]].mean().reset_index()
    m = pd.merge_asof(acc.sort_values("Timestamp"), gyr.sort_values("Timestamp"), on="Timestamp",
                      direction="nearest", tolerance=500).dropna()
    n = len(range(0, len(m) - 60, 30)); raw += n
    print(f"custom 'Stairs' file {f}: {n} raw windows")
custom_total = raw * (1 + 3)
print(f"custom Stairs after 3x augmentation: {raw} + {raw*3} = {custom_total}  (all of it Still/Jogging data, see MODEL_DIAGNOSIS)")
uci_stairs = None
if U and os.path.isdir(U):
    ys = np.concatenate([np.loadtxt(f"{U}/{s}/y_{s}.txt").astype(int) for s in ("train", "test")])
    uci_stairs = int(np.isin(ys, [2, 3]).sum())
    print(f"UCI HAR Stairs (labels 2+3, train+test): {uci_stairs} windows -> capped at 1500")
print("WISDM 'C' and Heterogeneity stairsup/down: datasets not available here; each capped at 1500 "
      "(assumes >= 1500 available, which the dataset sizes suggest but was not measured)")
pool = custom_total + 1500 + 1500 + 1500
print(f"Stairs pool before balancing (estimate): {pool};  share that is mislabelled custom data: {custom_total/pool:.0%}")
print("balance_dataset: Stairs -> 1500 (every other class -> 3500)")
print("train_test_split(test_size=0.2, stratified): Stairs ~1200 train / ~300 test; "
      "validation_split=0.2 -> ~960 fit / ~240 validation")

# ── 4. Signal comparison ──────────────────────────────────────────────────────
section("4. Signals (phone recordings at 20 Hz; per-window means)")


def dom_freq(w):
    mag = np.linalg.norm(w[:, :3], axis=1); mag = mag - mag.mean()
    f = np.fft.rfftfreq(len(mag), 1 / FS); p = np.abs(np.fft.rfft(mag)) ** 2
    band = (f >= 0.5) & (f <= 4.0)
    return f[band][np.argmax(p[band])]


rec_windows = {}
print(f"{'recording':10s} {'label':8s} {'n':>4s} {'|body acc|':>11s} {'|gyro|':>8s} {'raw acc std':>12s} {'step Hz':>8s}")
for r in RECORDINGS:
    segs, _ = parse_sensor_csv(open(path_for(r), "rb").read())
    _, W = windows_from_segments(segs); rec_windows[r["id"]] = W
    F = np.stack([preprocess_window(w) for w in W])
    print(f"{r['id']:10s} {r['label']:8s} {len(W):4d} {F[:, :, 6].mean():11.2f} {F[:, :, 7].mean():8.2f} "
          f"{np.mean([np.linalg.norm(w[:, :3], axis=1).std() for w in W]):12.2f} {np.median([dom_freq(w) for w in W]):8.2f}")

if U and os.path.isdir(U):
    from scipy.ndimage import zoom
    from scipy.signal import resample
    MAP = {1: "Walking", 2: "Stairs", 3: "Stairs"}
    y = np.loadtxt(f"{U}/test/y_test.txt").astype(int)
    ld = lambda ns: np.stack([np.loadtxt(f"{U}/test/Inertial Signals/{n}_test.txt") for n in ns], axis=-1)
    body = ld(["body_acc_x", "body_acc_y", "body_acc_z", "body_gyro_x", "body_gyro_y", "body_gyro_z"])
    total = ld(["total_acc_x", "total_acc_y", "total_acc_z", "body_gyro_x", "body_gyro_y", "body_gyro_z"])
    print("\nUCI HAR test windows, |body acc| channel after preprocessing:")
    for lab in (1, 2, 3):
        m = y == lab
        as_trained = np.mean([preprocess_window(zoom(w, (60 / 128, 1)))[:, 6].mean() for w in body[m][:300]])
        phone = total[m][:300].copy(); phone[:, :, :3] *= 9.80665
        phone_units = np.mean([preprocess_window(resample(w, 51))[:, 6].mean() for w in phone])  # 2.56 s at 20 Hz
        print(f"  label {lab} ({MAP[lab]:7s}) as trained: {as_trained:5.2f}   in phone units (m/s², 20 Hz): {phone_units:5.2f}")

# ── 5. Does inference suppress Stairs? ────────────────────────────────────────
section("5. Model and inference rules on the Stairs session (both files)")
W = np.concatenate([rec_windows["stairs1"], rec_windows["stairs2"]])
P = har.predict_probs(W)
si = C.index("Stairs")
rank = (-P).argsort(1).argsort(1)[:, si] + 1
print(f"windows: {len(W)};  model argmax == Stairs: {(P.argmax(1) == si).sum()}")
print(f"Stairs probability: mean {P[:, si].mean():.3f}, max {P[:, si].max():.3f}")
print("rank of Stairs among 6 classes:", {k: int((rank == k).sum()) for k in range(1, 7)})
rules, suppressed = {}, 0
for rid in ("stairs1", "stairs2"):
    sm = SessionSmoother()
    for w, p in zip(rec_windows[rid], har.predict_probs(rec_windows[rid])):
        out = har.postprocess(p, w, sm)
        rules[out["rule"]] = rules.get(out["rule"], 0) + 1
        if out["model_activity"] == "Stairs" and out["activity"] != "Stairs":
            suppressed += 1
print("post-processing rules fired:", rules)
print("windows where the model said Stairs but the dashboard did not:", suppressed)
for c in C:
    print(f"  mean prob {c:13s}: {P[:, C.index(c)].mean():.3f}")

# ── 6. Augmentation realism ───────────────────────────────────────────────────
section("6. Augmentation (prepare_data.augment_windows) on preprocessed 8-channel windows")
import importlib.util
spec = importlib.util.spec_from_file_location("prep", os.path.join(R, "prepare_data.py"))
prep = importlib.util.module_from_spec(spec)
cwd = os.getcwd(); os.chdir(R); spec.loader.exec_module(prep); os.chdir(cwd)
np.random.seed(0)
F = [preprocess_window(w) for w in rec_windows["walking1"][:50]]
for k, label in ((3, "custom augmentation, 3 copies (Stairs multiplier)"), (1, "balance_dataset oversampling (num_augments=1)")):
    A, _ = prep.augment_windows(F, ["x"] * len(F), num_augments=k)
    A = np.stack(A)
    print(f"{label}: {len(A)} windows; share with a NEGATIVE magnitude channel: {np.mean((A[:, :, 6:8] < -1e-6).any(axis=(1, 2))):.0%}")
