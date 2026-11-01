"""
Shared signal preprocessing for training (prepare_data.py) and inference (app.py).

The model was trained on windows produced by `preprocess_window`, so inference
must use exactly the same function — never a different filter implementation.
"""

import io

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

# ── Config (must match training) ──────────────────────────────────────────────
FS            = 20.0   # Hz
FRAME_SIZE    = 60     # samples per window (3 s at 20 Hz)
STEP_SIZE     = 30     # 50% overlap, same as training
RAW_CHANNELS  = 6      # accel x,y,z (m/s², incl. gravity) + gyro x,y,z (rad/s)
NUM_CHANNELS  = 8      # body accel x,y,z + gyro x,y,z + |body accel| + |gyro|

NYQ = 0.5 * FS
sos_noise   = butter(3, 5.0 / NYQ, btype='low', output='sos')
sos_gravity = butter(3, 0.3 / NYQ, btype='low', output='sos')


def preprocess_window(window_6d):
    """60×6 raw window -> 60×8 feature window (zero-phase filtering, per window)."""
    filtered = np.zeros_like(window_6d)
    for i in range(6):
        filtered[:, i] = sosfiltfilt(sos_noise, window_6d[:, i])
    accel = filtered[:, 0:3]
    gyro = filtered[:, 3:6]
    gravity = np.zeros_like(accel)
    for i in range(3):
        gravity[:, i] = sosfiltfilt(sos_gravity, accel[:, i])
    body_accel = accel - gravity
    accel_mag = np.linalg.norm(body_accel, axis=1, keepdims=True)
    gyro_mag = np.linalg.norm(gyro, axis=1, keepdims=True)
    return np.hstack((body_accel, gyro, accel_mag, gyro_mag))


def validate_window(data):
    """Return a float64 60×6 array or raise ValueError with a readable message."""
    try:
        arr = np.asarray(data, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("Window must be a list of numeric [ax, ay, az, gx, gy, gz] rows.")
    if arr.ndim != 2 or arr.shape[1] != RAW_CHANNELS:
        raise ValueError(f"Expected shape ({FRAME_SIZE}, {RAW_CHANNELS}), got {tuple(arr.shape)}.")
    if arr.shape[0] != FRAME_SIZE:
        raise ValueError(f"Expected {FRAME_SIZE} samples per window, got {arr.shape[0]}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Window contains NaN or infinite values.")
    return arr


def make_windows(samples, step=STEP_SIZE):
    """Slide a FRAME_SIZE window over an N×6 array. Returns (start_indices, windows)."""
    starts = list(range(0, len(samples) - FRAME_SIZE + 1, step))
    if not starts:
        return [], np.empty((0, FRAME_SIZE, RAW_CHANNELS))
    return starts, np.stack([samples[s:s + FRAME_SIZE] for s in starts])


# ── CSV parsing ───────────────────────────────────────────────────────────────
# Two formats are accepted:
#   1. Android sensor-logger format (the project's own recordings):
#        Timestamp,Sensor Type,Value1,Value2,Value3
#      Sensor Type 1 = accelerometer (m/s²), 4 = gyroscope (rad/s), timestamps in ms.
#   2. Wide format (also what the dashboard exports for raw sessions):
#        timestamp_ms(optional),ax,ay,az,gx,gy,gz
#      Without a timestamp column, rows are assumed to be sampled at 20 Hz.

WIDE_COLUMNS = ["ax", "ay", "az", "gx", "gy", "gz"]
MAX_GAP_MS = 1000  # never interpolate across a recording gap longer than this


def _resample_segments(accel, gyro, step_ms):
    """
    accel/gyro: DataFrames with columns [t, x, y, z] (t in ms).
    Resample both streams onto one 20 Hz grid, split wherever either stream has a gap.
    Returns a list of (start_ms, N×6 array).
    """
    accel = accel.groupby("t", as_index=False).mean().sort_values("t")
    gyro = gyro.groupby("t", as_index=False).mean().sort_values("t")
    t0, t1 = max(accel.t.min(), gyro.t.min()), min(accel.t.max(), gyro.t.max())
    if t1 <= t0:
        raise ValueError("Accelerometer and gyroscope readings do not overlap in time.")
    grid = np.arange(t0, t1 + step_ms / 2, step_ms)

    # A grid point is valid if it falls on a reading, or the readings on both sides are close together.
    valid = np.ones(len(grid), dtype=bool)
    for stream in (accel, gyro):
        t = stream.t.values
        idx = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
        on_reading = (np.abs(t[idx] - grid) < 1e-3) | (np.abs(t[idx - 1] - grid) < 1e-3)
        valid &= ((t[idx] - t[idx - 1]) <= MAX_GAP_MS) | on_reading

    values = np.column_stack(
        [np.interp(grid, accel.t.values, accel[c].values) for c in ("x", "y", "z")] +
        [np.interp(grid, gyro.t.values, gyro[c].values) for c in ("x", "y", "z")]
    )
    segments, start = [], None
    for i, ok in enumerate(np.append(valid, False)):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            segments.append((float(grid[start] - t0), values[start:i]))
            start = None
    return segments


def parse_sensor_csv(raw_bytes):
    """
    Parse an uploaded CSV into continuous 20 Hz segments.
    Returns (segments, info) where segments is a list of (start_sec, N×6 array).
    """
    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as e:
        raise ValueError(f"Could not read CSV: {e}")
    df.columns = [str(c).strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}
    step_ms = 1000.0 / FS

    if {"timestamp", "sensor type", "value1", "value2", "value3"} <= set(lower):
        cols = [lower[c] for c in ("timestamp", "sensor type", "value1", "value2", "value3")]
        df = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
        df.columns = ["t", "type", "x", "y", "z"]
        accel = df[df.type == 1][["t", "x", "y", "z"]]
        gyro = df[df.type == 4][["t", "x", "y", "z"]]
        if len(accel) < 2 or len(gyro) < 2:
            raise ValueError("CSV needs both accelerometer (type 1) and gyroscope (type 4) rows.")
        segments = _resample_segments(accel, gyro, step_ms)
        fmt = "android-logger"
    elif set(WIDE_COLUMNS) <= set(lower):
        cols = [lower[c] for c in WIDE_COLUMNS]
        t_name = next((lower[c] for c in ("timestamp_ms", "timestamp", "time_ms") if c in lower), None)
        df = df[cols + ([t_name] if t_name else [])].apply(pd.to_numeric, errors="coerce").dropna()
        if t_name and df[t_name].nunique() > 1:
            t = df[t_name]
            accel = pd.DataFrame({"t": t, "x": df[cols[0]], "y": df[cols[1]], "z": df[cols[2]]})
            gyro = pd.DataFrame({"t": t, "x": df[cols[3]], "y": df[cols[4]], "z": df[cols[5]]})
            segments = _resample_segments(accel, gyro, step_ms)
        else:
            segments = [(0.0, df[cols].values.astype(np.float64))]
        fmt = "wide"
    else:
        raise ValueError(
            "Unrecognised CSV columns. Use either 'Timestamp,Sensor Type,Value1,Value2,Value3' "
            "or 'timestamp_ms,ax,ay,az,gx,gy,gz'."
        )

    segments = [(t / 1000.0, seg) for t, seg in segments]  # ms -> s
    usable = sum(len(seg) for _, seg in segments)
    return segments, {
        "format": fmt,
        "samples": int(usable),
        "duration_sec": round(usable / FS, 2),
        "segments": len(segments),
    }


def windows_from_segments(segments, step=STEP_SIZE):
    """Window each continuous segment. Returns (start_times_sec, windows N×60×6)."""
    times, windows = [], []
    for start_sec, seg in segments:
        starts, w = make_windows(seg, step)
        times += [start_sec + s / FS for s in starts]
        if len(starts):
            windows.append(w)
    if not windows:
        return [], np.empty((0, FRAME_SIZE, RAW_CHANNELS))
    return times, np.concatenate(windows)
