"""
Train the v2 model on the windows built by ../prepare_data_v2.py.

Same architecture, loss and optimiser as train_model.py (create_model and focal_loss are
imported from it). What changes is the data handling:
  * splits come from prepare_data_v2.py (by subject / by recording, made before augmentation);
  * augmentation only on the training split, only for the few phone recordings, and only with
    physically valid transforms on the raw 6-channel signal (3-D rotation of accelerometer and
    gyroscope together, ±10 % amplitude, small sensor noise) before preprocessing — so the
    magnitude channels can never become negative;
  * the StandardScaler is fitted on the training split only;
  * no Stairs cap; class imbalance is handled by balanced class weights (as in train_model.py);
  * an explicit subject-wise validation split replaces validation_split=0.2.

Writes to backend/model_v2/ — the deployed backend/har_model.keras is never touched.

Usage (from backend/):  .venv/bin/python train_model_v2.py <windows.npz> [epochs]
"""

import json
import os
import sys
from collections import Counter

import numpy as np
from scipy.spatial.transform import Rotation
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from preprocessing import FRAME_SIZE, NUM_CHANNELS, preprocess_window
from train_model import ACTIVITY_NAMES_SORTED, BATCH_SIZE, create_model, save_artifacts

BASE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE, "model_v2")
SEED = 42
CAP_TRAIN_PER_SOURCE_CLASS = 2500   # keeps CPU training time reasonable; sampled across all subjects
CAP_VAL_PER_SOURCE_CLASS = 600
PHONE_AUG_COPIES = 10


def cap(idx, labels, sources, limit, rng):
    keep = []
    for key in sorted(set(zip(labels[idx], sources[idx]))):
        members = idx[(labels[idx] == key[0]) & (sources[idx] == key[1])]
        keep.extend(members if key[1] == "phone" or len(members) <= limit
                    else rng.choice(members, limit, replace=False))
    return np.array(sorted(keep))


def augment(w, rng):
    rot = Rotation.random(random_state=rng.integers(1 << 31)).as_matrix()
    out = np.empty_like(w)
    out[:, 0:3] = w[:, 0:3] @ rot.T
    out[:, 3:6] = w[:, 3:6] @ rot.T
    out *= rng.uniform(0.9, 1.1)
    out[:, 0:3] += rng.normal(0, 0.05, (len(w), 3))
    out[:, 3:6] += rng.normal(0, 0.01, (len(w), 3))
    return out


def table(title, labels, sources):
    print(f"\n{title}")
    srcs = sorted(set(sources))
    print(f"  {'class':14s}" + "".join(f"{s:>9s}" for s in srcs) + f"{'total':>9s}")
    for c in ACTIVITY_NAMES_SORTED:
        row = [int(((labels == c) & (sources == s)).sum()) for s in srcs]
        print(f"  {c:14s}" + "".join(f"{n:9d}" for n in row) + f"{sum(row):9d}")


def main():
    npz_path = sys.argv[1]
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    rng = np.random.default_rng(SEED)
    np.random.seed(SEED)
    import keras
    keras.utils.set_random_seed(SEED)
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau

    d = np.load(npz_path, allow_pickle=False)
    X, y, src, split = d["X"], d["y"], d["source"], d["split"]
    table("Loaded windows (all splits)", y, src)
    for s in ("train", "val", "test"):
        table(f"Split '{s}'", y[split == s], src[split == s])

    tr = cap(np.where(split == "train")[0], y, src, CAP_TRAIN_PER_SOURCE_CLASS, rng)
    va = cap(np.where(split == "val")[0], y, src, CAP_VAL_PER_SOURCE_CLASS, rng)
    table("Training windows after per-source cap (before augmentation)", y[tr], src[tr])

    Xtr, ytr, str_ = list(X[tr]), list(y[tr]), list(src[tr])
    for i in tr[src[tr] == "phone"]:
        for _ in range(PHONE_AUG_COPIES):
            Xtr.append(augment(X[i], rng)); ytr.append(y[i]); str_.append("phone_aug")
    Xtr, ytr, str_ = np.stack(Xtr), np.array(ytr), np.array(str_)
    table("Training windows used for fitting (after phone augmentation)", ytr, str_)

    print("\nPreprocessing ...", flush=True)
    Ftr = np.stack([preprocess_window(w.astype(np.float64)) for w in Xtr])
    Fva = np.stack([preprocess_window(w.astype(np.float64)) for w in X[va]])

    enc = LabelEncoder().fit(ACTIVITY_NAMES_SORTED)
    Ytr, Yva = enc.transform(ytr), enc.transform(y[va])
    scaler = StandardScaler().fit(Ftr.reshape(-1, NUM_CHANNELS))     # training split only
    sc = lambda F: scaler.transform(F.reshape(-1, NUM_CHANNELS)).reshape(F.shape)
    Ftr, Fva = sc(Ftr), sc(Fva)

    cw = compute_class_weight("balanced", classes=np.unique(Ytr), y=Ytr)
    class_weight = dict(zip(np.unique(Ytr).tolist(), cw.tolist()))
    print("Class weights:", {enc.classes_[k]: round(v, 3) for k, v in class_weight.items()})

    model = create_model(input_shape=(FRAME_SIZE, NUM_CHANNELS), num_classes=len(ACTIVITY_NAMES_SORTED))
    hist = model.fit(
        Ftr, Ytr, validation_data=(Fva, Yva), batch_size=BATCH_SIZE, epochs=epochs, shuffle=True,
        class_weight=class_weight, verbose=2,
        callbacks=[EarlyStopping(monitor="val_accuracy", patience=15, restore_best_weights=True, verbose=1),
                   ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1)],
    )
    save_artifacts(model, scaler, enc, OUTPUT_DIR)
    best = int(np.argmax(hist.history["val_accuracy"]))
    summary = {
        "source_npz": os.path.basename(npz_path),
        "epochs_run": len(hist.history["loss"]), "best_epoch": best + 1,
        "best_val_accuracy": round(float(hist.history["val_accuracy"][best]), 4),
        "train_counts": {c: int(n) for c, n in Counter(ytr).items()},
        "val_counts": {c: int(n) for c, n in Counter(y[va]).items()},
        "class_weight": {enc.classes_[k]: round(v, 4) for k, v in class_weight.items()},
        "caps": {"train_per_source_class": CAP_TRAIN_PER_SOURCE_CLASS,
                 "val_per_source_class": CAP_VAL_PER_SOURCE_CLASS, "phone_aug_copies": PHONE_AUG_COPIES},
        "history": {k: [round(float(v), 4) for v in vals] for k, vals in hist.history.items()},
    }
    with open(os.path.join(OUTPUT_DIR, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nBest epoch {best + 1}, validation accuracy {summary['best_val_accuracy']}")


if __name__ == "__main__":
    main()
