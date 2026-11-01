"""
Corrected training-data preparation (v2). The original prepare_data.py is kept unchanged
because it documents how the deployed model (backend/har_model.keras) was built.

Differences from prepare_data.py, each fixing a problem found in MODEL_DIAGNOSIS.md:
  * Every source is converted to the live-app input format BEFORE windowing:
    accelerometer in m/s² including gravity, gyroscope in rad/s, resampled on its own
    timestamps to exactly 20 Hz, windows of 60 samples (3 s) with a 30-sample step.
      - UCI HAR: total_acc (g, incl. gravity) × 9.80665 instead of body_acc (g, gravity removed);
        the 2.56 s windows are stitched back into continuous 50 Hz signals and resampled to 20 Hz
        instead of squeezing 128 samples into 60.
      - WISDM and Heterogeneity: accelerometer and gyroscope are aligned by timestamp per
        subject/device instead of by row number; no fixed every-10th-row decimation.
      - Phone recordings: the same parse_sensor_csv() the backend uses for uploaded CSVs.
  * No Stairs cap below the other classes; no mislabelled duplicate files.
  * Raw windows are stored with subject / recording ids and a split assigned BY SUBJECT
    (public data) or BY RECORDING (phone data), before any augmentation.
    Augmentation and scaling happen later, on the training split only (backend/train_model_v2.py).

Usage:  backend/.venv/bin/python prepare_data_v2.py <datasets_dir> <output.npz>
  <datasets_dir> must contain:
    uci_har/UCI HAR Dataset/
    wisdm/wisdm-dataset/raw/phone/{accel,gyro}/
    heterogeneity/Phones_accelerometer.csv, heterogeneity/Phones_gyroscope.csv
"""

import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "backend"))
from preprocessing import FRAME_SIZE, FS, STEP_SIZE, _resample_segments, make_windows, parse_sensor_csv  # noqa: E402
from recordings import RECORDINGS, path_for  # noqa: E402

G = 9.80665
SEED = 42
STEP_MS = 1000.0 / FS

# Original dataset labels -> the six classes (same mapping as prepare_data.py).
WISDM_MAP = {"A": "Walking", "B": "Jogging", "C": "Stairs", "D": "Still", "E": "Still",
             "F": "Hand Activity", "G": "Hand Activity", "Q": "Hand Activity", "R": "Hand Activity",
             "S": "Hand Activity", "M": "Sports", "O": "Sports", "P": "Sports"}
HET_MAP = {"walk": "Walking", "sit": "Still", "stand": "Still", "stairsup": "Stairs",
           "stairsdown": "Stairs", "bike": "Sports"}
UCI_MAP = {1: "Walking", 2: "Stairs", 3: "Stairs", 4: "Still", 5: "Still", 6: "Still"}

# Phone recordings: the Stairs session (both files) and one Walking recording are held out.
PHONE_TEST_IDS = {"stairs1", "stairs2", "walking2"}

# Subject-wise splits (fixed seed, decided before looking at any results).
N_TEST_WISDM, N_VAL_WISDM = 10, 5
N_TEST_HET, N_VAL_HET = 2, 1
N_VAL_UCI = 3   # taken from the official UCI training subjects; official test subjects stay test


def split_subjects(subjects, n_test, n_val, seed):
    s = sorted(subjects)
    rng = np.random.default_rng(seed)
    rng.shuffle(s)
    return {**{x: "test" for x in s[:n_test]}, **{x: "val" for x in s[n_test:n_test + n_val]},
            **{x: "train" for x in s[n_test + n_val:]}}


class Collector:
    def __init__(self):
        self.X, self.y, self.source, self.subject, self.split, self.orig = [], [], [], [], [], []

    def add(self, segment, label, source, subject, split, orig):
        _, w = make_windows(segment, STEP_SIZE)
        for win in w:
            self.X.append(win.astype(np.float32))
            self.y.append(label); self.source.append(source); self.subject.append(str(subject))
            self.split.append(split); self.orig.append(str(orig))
        return len(w)


# ── UCI HAR ───────────────────────────────────────────────────────────────────
def load_uci(base, out):
    print("UCI HAR ...", flush=True)
    for part in ("train", "test"):
        d = os.path.join(base, part)
        y = np.loadtxt(f"{d}/y_{part}.txt").astype(int)
        subj = np.loadtxt(f"{d}/subject_{part}.txt").astype(int)
        names = ["total_acc_x", "total_acc_y", "total_acc_z", "body_gyro_x", "body_gyro_y", "body_gyro_z"]
        W = np.stack([np.loadtxt(f"{d}/Inertial Signals/{n}_{part}.txt") for n in names], axis=-1)
        W[:, :, :3] *= G                                   # g -> m/s² (gravity included)
        if part == "train":
            val_subj = split_subjects(set(subj), 0, N_VAL_UCI, SEED)
        # Stitch consecutive 50%-overlapping windows of the same subject and activity.
        runs, cur = [], None
        for i in range(len(y)):
            contiguous = (cur is not None and subj[i] == subj[i - 1] and y[i] == y[i - 1]
                          and np.allclose(W[i - 1, 64:], W[i, :64], atol=1e-4))
            if contiguous:
                cur["sig"].append(W[i, 64:])
            else:
                if cur:
                    runs.append(cur)
                cur = {"sig": [W[i]], "label": y[i], "subject": subj[i]}
        runs.append(cur)
        for r in runs:
            sig50 = np.concatenate(r["sig"])
            sig20 = resample_poly(sig50, 2, 5, axis=0)     # 50 Hz -> 20 Hz with anti-alias filter
            split = "test" if part == "test" else val_subj[r["subject"]]
            out.add(sig20, UCI_MAP[r["label"]], "uci", f"uci{r['subject']}", split, r["label"])


# ── Phone recordings in the repository ────────────────────────────────────────
def load_phone(out):
    print("Phone recordings ...", flush=True)
    for rec in RECORDINGS:
        with open(path_for(rec), "rb") as f:
            segments, _ = parse_sensor_csv(f.read())
        split = "test" if rec["id"] in PHONE_TEST_IDS else "train"
        for _, seg in segments:
            out.add(seg, rec["label"], "phone", rec["id"], split, rec["file"])


def resample_pair(acc, gyr):
    """acc/gyr: DataFrames [t (ms), x, y, z] -> list of continuous 20 Hz N×6 segments."""
    return [seg for _, seg in _resample_segments(acc, gyr, STEP_MS) if len(seg) >= FRAME_SIZE]


# ── WISDM (phone only) ────────────────────────────────────────────────────────
def load_wisdm(base, out, skipped):
    print("WISDM ...", flush=True)
    cols = ["subject", "code", "t", "x", "y", "z"]
    read = lambda p: pd.read_csv(p, header=None, names=cols, lineterminator="\n",
                                 converters={"z": lambda v: float(v.rstrip(";")) if v.strip(";") else np.nan})
    subjects = sorted(int(f.split("_")[1]) for f in os.listdir(f"{base}/accel") if f.endswith(".txt"))
    splits = split_subjects(subjects, N_TEST_WISDM, N_VAL_WISDM, SEED)
    for s in subjects:
        acc = read(f"{base}/accel/data_{s}_accel_phone.txt").dropna()
        gyr = read(f"{base}/gyro/data_{s}_gyro_phone.txt").dropna()
        for code, label in WISDM_MAP.items():
            a = acc[acc.code == code][["t", "x", "y", "z"]].copy(); g = gyr[gyr.code == code][["t", "x", "y", "z"]].copy()
            if len(a) < FRAME_SIZE or len(g) < FRAME_SIZE:
                skipped[("wisdm", "missing stream")] += 1; continue
            a["t"] = a.t / 1e6; g["t"] = g.t / 1e6                       # ns -> ms
            try:
                segs = resample_pair(a, g)
            except ValueError:
                skipped[("wisdm", "accel/gyro do not overlap in time")] += 1; continue
            for seg in segs:
                out.add(seg, label, "wisdm", f"wisdm{s}", splits[s], code)


# ── Heterogeneity (phones only) ───────────────────────────────────────────────
def load_het(base, out, skipped):
    print("Heterogeneity (large CSVs, this takes a few minutes) ...", flush=True)
    usecols = ["Arrival_Time", "Creation_Time", "x", "y", "z", "User", "Device", "gt"]
    dtypes = {"x": "float32", "y": "float32", "z": "float32", "User": "category", "Device": "category", "gt": "category"}
    acc = pd.read_csv(f"{base}/Phones_accelerometer.csv", usecols=usecols, dtype=dtypes)
    gyr = pd.read_csv(f"{base}/Phones_gyroscope.csv", usecols=usecols, dtype=dtypes)
    acc = acc[acc["gt"].isin(HET_MAP)]; gyr = gyr[gyr["gt"].isin(HET_MAP)]   # drops "null"
    gyr_groups = {k: v for k, v in gyr.groupby(["User", "Device", "gt"], observed=True)}
    splits = split_subjects(set(acc.User.astype(str)), N_TEST_HET, N_VAL_HET, SEED)
    for (user, dev, gt), a in acc.groupby(["User", "Device", "gt"], observed=True):
        g = gyr_groups.get((user, dev, gt))
        if g is None or len(a) < FRAME_SIZE or len(g) < FRAME_SIZE:
            skipped[("het", "missing stream")] += 1; continue
        segs = None
        for tcol, scale in (("Creation_Time", 1e6), ("Arrival_Time", 1.0)):   # sensor clock first
            try:
                A = pd.DataFrame({"t": a[tcol].values / scale, "x": a.x.values, "y": a.y.values, "z": a.z.values})
                Gy = pd.DataFrame({"t": g[tcol].values / scale, "x": g.x.values, "y": g.y.values, "z": g.z.values})
                segs = resample_pair(A, Gy)
                if segs:
                    break
            except ValueError:
                continue
        if not segs:
            skipped[("het", "no aligned segment")] += 1; continue
        for seg in segs:
            out.add(seg, HET_MAP[gt], "het", f"het_{user}", splits[str(user)], f"{gt}|{dev}")


def main():
    data_dir, out_path = sys.argv[1], sys.argv[2]
    out, skipped = Collector(), Counter()
    load_phone(out)
    load_uci(os.path.join(data_dir, "uci_har", "UCI HAR Dataset"), out)
    load_wisdm(os.path.join(data_dir, "wisdm", "wisdm-dataset", "raw", "phone"), out, skipped)
    load_het(os.path.join(data_dir, "heterogeneity"), out, skipped)
    y, src, split = np.array(out.y), np.array(out.source), np.array(out.split)
    print("\nWindows per class / source / split (raw 60×6 at 20 Hz):")
    for c in sorted(set(y)):
        print(f"  {c:14s}" + "  ".join(f"{s}:{sp[0]}={int(((y == c) & (src == s) & (split == sp)).sum())}"
                                     for s in ("phone", "uci", "wisdm", "het") for sp in ("train", "val", "test")))
    print("Skipped groups:", dict(skipped))
    subj = np.array(out.subject)
    for sp in ("train", "val", "test"):
        print(f"  {sp:5s} subjects:", sorted(set(subj[split == sp])))
    overlap = set(subj[split == "train"]) & set(subj[split == "test"])
    print("Subjects in BOTH train and test:", overlap or "none")
    np.savez_compressed(out_path, X=np.stack(out.X), y=y, source=src, subject=subj, split=split,
                        orig=np.array(out.orig))
    print(f"Saved {len(y)} windows -> {out_path}")


if __name__ == "__main__":
    main()
