import os
import sys
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from scipy import stats
import pickle
from collections import defaultdict

# Configuration
WINDOW_SIZE = 60
STEP_SIZE = 30
CHANNELS = 8
TARGET_PER_CLASS = 3500  # Balanced target per class

ACTIVITY_MERGE = {
    "A": "Walking", "B": "Jogging", "C": "Stairs",
    "D": "Still", "E": "Still",
    "F": "Hand Activity", "G": "Hand Activity", "Q": "Hand Activity",
    "M": "Sports", "O": "Sports", "P": "Sports",
    "R": "Hand Activity", "S": "Hand Activity",
    "walk": "Walking", "sit": "Still", "stand": "Still",
    "stairsup": "Stairs", "stairsdown": "Stairs", "bike": "Sports",
    "1": "Walking", "2": "Stairs", "3": "Stairs",
    "4": "Still", "5": "Still", "6": "Still"
}

# Same function the backend uses at inference time (keeps training and serving identical).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))
from preprocessing import preprocess_window  # noqa: E402

def clean_wisdm(file_path):
    try:
        records = []
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().rstrip(';').split(',')
                if len(parts) >= 6:
                    try:
                        records.append([parts[1], float(parts[3]), float(parts[4]), float(parts[5])])
                    except ValueError:
                        pass
        return pd.DataFrame(records, columns=["ActivityCode", "X", "Y", "Z"])
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return pd.DataFrame()

def extract_wisdm():
    print("Processing WISDM...", flush=True)
    accel_dir = "wisdm_data/extracted/wisdm-dataset/raw/phone/accel"
    gyro_dir = "wisdm_data/extracted/wisdm-dataset/raw/phone/gyro"
    all_frames, all_labels = [], []
    for sid in range(1600, 1651):
        a_path = f"{accel_dir}/data_{sid}_accel_phone.txt"
        g_path = f"{gyro_dir}/data_{sid}_gyro_phone.txt"
        if not os.path.exists(a_path) or not os.path.exists(g_path):
            continue
        df_a = clean_wisdm(a_path)
        df_g = clean_wisdm(g_path)
        if df_a.empty or df_g.empty:
            continue
        for act in df_a["ActivityCode"].unique():
            a_act = df_a[df_a["ActivityCode"] == act]
            g_act = df_g[df_g["ActivityCode"] == act]
            min_len = min(len(a_act), len(g_act))
            if min_len < WINDOW_SIZE: continue
            merged = np.column_stack((
                a_act["X"].values[:min_len], a_act["Y"].values[:min_len], a_act["Z"].values[:min_len],
                g_act["X"].values[:min_len], g_act["Y"].values[:min_len], g_act["Z"].values[:min_len]
            ))
            label = ACTIVITY_MERGE.get(act, "Unknown")
            if label == "Unknown": continue
            for i in range(0, min_len - WINDOW_SIZE, STEP_SIZE):
                window = merged[i:i+WINDOW_SIZE]
                window = preprocess_window(window)
                all_frames.append(window)
                all_labels.append(label)
    print(f"  WISDM: {len(all_frames)} windows", flush=True)
    return all_frames, all_labels

def extract_heterogeneity():
    print("Processing Heterogeneity...", flush=True)
    import zipfile
    all_frames, all_labels = [], []
    zip_path = "heterogeneity+activity+recognition/Activity recognition exp.zip"
    if not os.path.exists(zip_path):
        print("  Heterogeneity dataset not found, skipping.", flush=True)
        return [], []
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open('Activity recognition exp/Phones_accelerometer.csv') as f:
            df_a = pd.read_csv(f)
        with z.open('Activity recognition exp/Phones_gyroscope.csv') as f:
            df_g = pd.read_csv(f)
    print(f"  Loaded accel: {len(df_a)} rows, gyro: {len(df_g)} rows", flush=True)
    df_a = df_a.iloc[::10].reset_index(drop=True)
    df_g = df_g.iloc[::10].reset_index(drop=True)
    users = df_a["User"].unique()
    for u in users:
        u_a = df_a[df_a["User"] == u]
        u_g = df_g[df_g["User"] == u]
        for gt in u_a["gt"].unique():
            act_a = u_a[u_a["gt"] == gt]
            act_g = u_g[u_g["gt"] == gt]
            min_len = min(len(act_a), len(act_g))
            if min_len < WINDOW_SIZE: continue
            merged = np.column_stack((
                act_a["x"].values[:min_len], act_a["y"].values[:min_len], act_a["z"].values[:min_len],
                act_g["x"].values[:min_len], act_g["y"].values[:min_len], act_g["z"].values[:min_len]
            ))
            label = ACTIVITY_MERGE.get(gt, "Unknown")
            if label == "Unknown": continue
            for i in range(0, min_len - WINDOW_SIZE, STEP_SIZE):
                window = merged[i:i+WINDOW_SIZE]
                window = preprocess_window(window)
                all_frames.append(window)
                all_labels.append(label)
    print(f"  Heterogeneity: {len(all_frames)} windows", flush=True)
    return all_frames, all_labels

def extract_uci_har():
    print("Processing UCI HAR...", flush=True)
    base = "UCI HAR Dataset (1)/UCI HAR Dataset"
    all_frames, all_labels = [], []
    for split in ["train", "test"]:
        labels_path = f"{base}/{split}/y_{split}.txt"
        if not os.path.exists(labels_path): continue
        with open(labels_path, "r") as f:
            y = [line.strip() for line in f.readlines()]
        channels = []
        c_names = ["body_acc_x", "body_acc_y", "body_acc_z", "body_gyro_x", "body_gyro_y", "body_gyro_z"]
        for c in c_names:
            c_path = f"{base}/{split}/Inertial Signals/{c}_{split}.txt"
            with open(c_path, "r") as f:
                data = [[float(v) for v in line.split()] for line in f.readlines()]
                channels.append(data)
        N = len(y)
        for i in range(N):
            label = ACTIVITY_MERGE.get(y[i], "Unknown")
            if label == "Unknown": continue
            window_128 = np.column_stack([channels[c][i] for c in range(6)])
            window_60 = zoom(window_128, (60/128, 1))
            window_processed = preprocess_window(window_60)
            all_frames.append(window_processed)
            all_labels.append(label)
    print(f"  UCI HAR: {len(all_frames)} windows", flush=True)
    return all_frames, all_labels


def extract_custom_csv(csv_path, activity_label):
    print(f"  Processing: {csv_path} -> {activity_label}", flush=True)
    if not os.path.exists(csv_path):
        print(f"    File not found: {csv_path}", flush=True)
        return [], []
    df = pd.read_csv(csv_path)
    accel = df[df['Sensor Type'] == 1][['Timestamp', 'Value1', 'Value2', 'Value3']].copy()
    gyro = df[df['Sensor Type'] == 4][['Timestamp', 'Value1', 'Value2', 'Value3']].copy()
    accel = accel.groupby('Timestamp').mean().reset_index().sort_values('Timestamp')
    gyro = gyro.groupby('Timestamp').mean().reset_index().sort_values('Timestamp')
    print(f"    Accel: {len(accel)}, Gyro: {len(gyro)} readings", flush=True)
    if len(accel) < WINDOW_SIZE or len(gyro) < WINDOW_SIZE:
        return [], []
    accel = accel.rename(columns={'Value1': 'ax', 'Value2': 'ay', 'Value3': 'az'})
    gyro = gyro.rename(columns={'Value1': 'gx', 'Value2': 'gy', 'Value3': 'gz'})
    merged = pd.merge_asof(
        accel.sort_values('Timestamp'), gyro.sort_values('Timestamp'),
        on='Timestamp', direction='nearest', tolerance=500
    ).dropna()
    print(f"    Paired: {len(merged)} readings", flush=True)
    if len(merged) < WINDOW_SIZE:
        return [], []
    data_6ch = merged[['ax', 'ay', 'az', 'gx', 'gy', 'gz']].values
    all_frames, all_labels = [], []
    for i in range(0, len(data_6ch) - WINDOW_SIZE, STEP_SIZE):
        window = data_6ch[i:i + WINDOW_SIZE]
        window_processed = preprocess_window(window)
        all_frames.append(window_processed)
        all_labels.append(activity_label)
    print(f"    Windows: {len(all_frames)}", flush=True)
    return all_frames, all_labels


def augment_windows(frames, labels, num_augments=30):
    """Create augmented copies via diverse, advanced techniques."""
    aug_frames, aug_labels = [], []
    for frame, label in zip(frames, labels):
        frame = np.array(frame)
        for i in range(num_augments):
            aug = frame.copy()
            noise_level = np.random.uniform(0.01, 0.05)
            aug += np.random.normal(0, noise_level, aug.shape)
            scale = np.random.uniform(0.90, 1.10)
            aug *= scale
            shift = np.random.randint(0, 8)
            if shift > 0:
                aug = np.roll(aug, shift, axis=0)
            if i % 3 == 0:
                mid = aug.shape[0] // 2
                warp_factor = np.random.uniform(0.8, 1.2)
                first_half = zoom(aug[:mid], (warp_factor, 1))
                second_half = zoom(aug[mid:], ((2.0 - warp_factor), 1))
                combined = np.vstack((first_half, second_half))
                if combined.shape[0] != aug.shape[0]:
                    combined = zoom(combined, (aug.shape[0] / combined.shape[0], 1))
                aug = combined
            if i % 5 == 0:
                perm = np.random.permutation(3)
                aug[:, :3] = aug[:, perm]
                perm2 = np.random.permutation(3)
                aug[:, 3:6] = aug[:, 3 + perm2]
            if i % 4 == 0:
                ch = np.random.randint(0, aug.shape[1])
                aug[:, ch] *= -1
            aug_frames.append(aug)
            aug_labels.append(label)
    return aug_frames, aug_labels


def balance_dataset(frames, labels, target_per_class=TARGET_PER_CLASS):
    """Globally balance: oversample minority, undersample majority."""
    print(f"\n  Balancing dataset -> target {target_per_class} per class...", flush=True)
    class_buckets = defaultdict(list)
    for f, l in zip(frames, labels):
        class_buckets[l].append(f)

    balanced_frames, balanced_labels = [], []
    for label, bucket in sorted(class_buckets.items()):
        current = len(bucket)
        
        # Explicitly reduce Stairs representation in the dataset
        current_target = 1500 if label == "Stairs" else target_per_class
        
        if current == 0:
            print(f"    {label:15s}: SKIPPED (no data)", flush=True)
            continue
        if current >= current_target:
            indices = np.random.choice(current, current_target, replace=False)
            selected = [bucket[i] for i in indices]
            balanced_frames.extend(selected)
            balanced_labels.extend([label] * current_target)
            print(f"    {label:15s}: {current:5d} -> {current_target} (undersampled)", flush=True)
        else:
            balanced_frames.extend(bucket)
            balanced_labels.extend([label] * current)
            needed = current_target - current
            # Oversample by repeating + augmenting
            aug_sources = []
            aug_source_labels = []
            while len(aug_sources) < needed:
                idx = np.random.randint(0, current)
                aug_sources.append(bucket[idx])
                aug_source_labels.append(label)
            af, al = augment_windows(aug_sources[:needed], aug_source_labels[:needed], num_augments=1)
            balanced_frames.extend(af[:needed])
            balanced_labels.extend(al[:needed])
            print(f"    {label:15s}: {current:5d} -> {current + needed} (oversampled)", flush=True)
    return balanced_frames, balanced_labels


def extract_all_custom():
    """Extract with CLASS-AWARE augmentation (less for Stairs, more for minority)."""
    print("\nProcessing Custom Mobile Data...", flush=True)
    custom_files = {
        "still data.csv": "Still",
        "jogging.csv": "Jogging",
        # NOTE: sensor_recording_20260424_131838.csv is byte-identical to "still data.csv" and
        # sensor_recording_20260424_184254.csv is identical to "jogging.csv". Both were previously
        # ALSO labelled "Stairs", which taught the model that still/jogging data can be Stairs.
        # sensor_recording_20260424_184845.csv is identical to eating.csv (label unclear) -> excluded.
        "sensor_recording_20260426_145530.csv": "Stairs",
        "sensor_recording_20260426_145547.csv": "Stairs",
        "sensor_recording_20260426_162439.csv": "Walking",
        "sensor_recording_20260427_174501.csv": "Walking",
    }
    AUG_MULTIPLIERS = {
        "Stairs": 3, "Still": 30, "Jogging": 25, "Walking": 25,
        "Hand Activity": 30, "Sports": 30,
    }
    class_raw = defaultdict(list)
    class_labels_map = defaultdict(list)
    for csv_file, activity in custom_files.items():
        f, l = extract_custom_csv(csv_file, activity)
        if f:
            class_raw[activity].extend(f)
            class_labels_map[activity].extend(l)

    all_frames, all_labels = [], []
    for activity, raw_frames in class_raw.items():
        raw_labels = class_labels_map[activity]
        all_frames.extend(raw_frames)
        all_labels.extend(raw_labels)
        num_aug = AUG_MULTIPLIERS.get(activity, 10)
        af, al = augment_windows(raw_frames, raw_labels, num_augments=num_aug)
        all_frames.extend(af)
        all_labels.extend(al)
        print(f"    {activity}: {len(raw_frames)} raw + {len(af)} aug ({num_aug}x)", flush=True)
    print(f"  Total custom: {len(all_frames)} windows", flush=True)
    return all_frames, all_labels


MAX_PUBLIC_PER_CLASS = 1500


def cap_dataset_per_class(frames, labels, max_per_class=500):
    class_buckets = defaultdict(list)
    for f, l in zip(frames, labels):
        class_buckets[l].append(f)
    capped_frames, capped_labels = [], []
    for label, bucket in class_buckets.items():
        np.random.shuffle(bucket)
        selected = bucket[:max_per_class]
        capped_frames.extend(selected)
        capped_labels.extend([label] * len(selected))
    return capped_frames, capped_labels


if __name__ == "__main__":
    print("=" * 60, flush=True)
    print("  PREPARING HAR TRAINING DATA (BALANCED, OPTIMIZED)", flush=True)
    print("=" * 60, flush=True)

    frames, labels = [], []

    f4, l4 = extract_all_custom()
    frames.extend(f4); labels.extend(l4)
    print(f"\n  Custom data: {len(f4)} windows", flush=True)

    f1, l1 = extract_uci_har()
    f1, l1 = cap_dataset_per_class(f1, l1, max_per_class=MAX_PUBLIC_PER_CLASS)
    frames.extend(f1); labels.extend(l1)
    print(f"  UCI HAR (capped {MAX_PUBLIC_PER_CLASS}): {len(f1)} windows", flush=True)

    f2, l2 = extract_heterogeneity()
    f2, l2 = cap_dataset_per_class(f2, l2, max_per_class=MAX_PUBLIC_PER_CLASS)
    frames.extend(f2); labels.extend(l2)
    print(f"  Heterogeneity (capped {MAX_PUBLIC_PER_CLASS}): {len(f2)} windows", flush=True)

    f3, l3 = extract_wisdm()
    f3, l3 = cap_dataset_per_class(f3, l3, max_per_class=MAX_PUBLIC_PER_CLASS)
    frames.extend(f3); labels.extend(l3)
    print(f"  WISDM (capped {MAX_PUBLIC_PER_CLASS}): {len(f3)} windows", flush=True)

    pre_labels = np.array(labels)
    print(f"\n{'=' * 60}", flush=True)
    print("  PRE-BALANCE class distribution:", flush=True)
    unique, counts = np.unique(pre_labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    {u:15s}: {c:6d} ({c/len(pre_labels)*100:.1f}%)", flush=True)

    # GLOBAL CLASS BALANCING
    frames, labels = balance_dataset(frames, labels, target_per_class=TARGET_PER_CLASS)

    X = np.array(frames)
    y = np.array(labels)

    print(f"\n{'=' * 60}", flush=True)
    print(f"  FINAL (BALANCED) dataset:", flush=True)
    print(f"  Total Frames: {X.shape}", flush=True)
    print(f"  Total Labels: {y.shape}", flush=True)

    unique, counts = np.unique(y, return_counts=True)
    print("\n  Class distribution:", flush=True)
    for u, c in zip(unique, counts):
        print(f"    {u:15s}: {c:6d} ({c/len(y)*100:.1f}%)", flush=True)

    max_c = max(counts); min_c = min(counts)
    ratio = max_c / min_c if min_c > 0 else float('inf')
    print(f"\n  Balance ratio (max/min): {ratio:.2f}x", flush=True)

    np.save("X_all.npy", X)
    np.save("y_all.npy", y)
    print("\nSaved to X_all.npy and y_all.npy", flush=True)
    print("=" * 60, flush=True)
