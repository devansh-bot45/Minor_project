"""
Model loading and inference for the Conv1D + BiLSTM + Temporal Attention HAR model.

Pipeline for one 60×6 raw window:
  preprocess_window (same as training) -> StandardScaler -> model.predict
  -> fixed temperature softening -> per-session EMA -> threshold / rules -> majority vote
"""

import collections
import os
import pickle
import warnings

import numpy as np

from preprocessing import FRAME_SIZE, NUM_CHANNELS, preprocess_window

# ── Post-processing config (unchanged from the original backend) ──────────────
CONFIDENCE_THRESHOLD = 0.40
SMOOTHING_WINDOW     = 5
TEMPERATURE          = 1.5   # fixed value, not fitted on validation data
EMA_ALPHA            = 0.4   # weight of the newest window in the moving average

# A phone lying on a table: almost no variance in the raw signals.
STILL_VAR_THRESHOLD = 0.12

# Intensity-based gating (total_var = accel_var + gyro_var on raw signals).
# Stops slight wrist movements from being labelled Stairs / Jogging.
INTENSITY_SLIGHT   = 0.8
INTENSITY_STAIRS   = 2.0
INTENSITY_WALK_JOG = 5.0
ACTIVITIES_SLIGHT   = {"Still", "Hand Activity"}
ACTIVITIES_STAIRS   = {"Still", "Hand Activity", "Stairs"}
ACTIVITIES_WALK_JOG = {"Still", "Hand Activity", "Stairs", "Walking", "Jogging"}

UNCERTAIN = "Uncertain"


def _temperature_scale(probs, temperature=TEMPERATURE):
    logits = np.log(probs + 1e-8) / temperature
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def motion_variance(raw_window):
    """Sum of per-axis variances of raw accel and gyro (used by the rules)."""
    return float(np.sum(np.var(raw_window[:, 0:3], axis=0))), float(np.sum(np.var(raw_window[:, 3:6], axis=0)))


class SessionSmoother:
    """Temporal smoothing state for ONE recording session (never shared between users)."""

    def __init__(self):
        self.ema = None
        self.recent = collections.deque(maxlen=SMOOTHING_WINDOW)

    def update_probs(self, probs):
        self.ema = probs.copy() if self.ema is None else EMA_ALPHA * probs + (1 - EMA_ALPHA) * self.ema
        return self.ema

    def vote(self, activity):
        self.recent.append(activity)
        return collections.Counter(self.recent).most_common(1)[0][0]


class HARModel:
    def __init__(self, base_dir):
        import keras  # imported here so the rest of the backend can be tested without loading TF

        # compile=False: the focal loss is only needed for training, not for inference.
        self.model = keras.models.load_model(os.path.join(base_dir, "har_model.keras"), compile=False)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # sklearn version-mismatch warnings for simple objects
            with open(os.path.join(base_dir, "activity_names.pkl"), "rb") as f:
                names = pickle.load(f)
            with open(os.path.join(base_dir, "label_encoder.pkl"), "rb") as f:
                encoder = pickle.load(f)
            with open(os.path.join(base_dir, "scaler.pkl"), "rb") as f:
                self.scaler = pickle.load(f)

        self.classes = [str(names[i]) for i in range(len(names))]

        # Fail loudly if the artifacts do not belong together.
        if list(map(str, encoder.classes_)) != self.classes:
            raise RuntimeError(f"label_encoder {list(encoder.classes_)} != activity_names {self.classes}")
        if self.model.output_shape[-1] != len(self.classes):
            raise RuntimeError(f"Model outputs {self.model.output_shape[-1]} classes, expected {len(self.classes)}")
        if tuple(self.model.input_shape[1:]) != (FRAME_SIZE, NUM_CHANNELS):
            raise RuntimeError(f"Model input {self.model.input_shape}, expected (None, {FRAME_SIZE}, {NUM_CHANNELS})")
        if self.scaler.n_features_in_ != NUM_CHANNELS:
            raise RuntimeError("Scaler was not fitted on 8 channels")

        self.input_shape = [FRAME_SIZE, NUM_CHANNELS]
        self.param_count = int(self.model.count_params())

    # ── Raw model output ──────────────────────────────────────────────────────
    def predict_probs(self, raw_windows):
        """N×60×6 raw windows -> N×C temperature-softened class probabilities."""
        feats = np.stack([preprocess_window(w) for w in raw_windows])
        n = feats.shape[0]
        feats = self.scaler.transform(feats.reshape(-1, NUM_CHANNELS)).reshape(n, FRAME_SIZE, NUM_CHANNELS)
        raw = self.model.predict(feats, verbose=0, batch_size=128)
        return _temperature_scale(raw)

    # ── Rules + smoothing ─────────────────────────────────────────────────────
    def postprocess(self, probs, raw_window, smoother=None, apply_rules=True):
        """
        Turn one probability vector into a prediction dict.
        `model_activity` is always the pure model argmax, so the UI and the
        evaluation can show when a rule changed the result.
        """
        model_idx = int(np.argmax(probs))
        model_activity = self.classes[model_idx]
        accel_var, gyro_var = motion_variance(raw_window)
        rule = None

        decision_probs = smoother.update_probs(probs) if smoother else probs
        idx = int(np.argmax(decision_probs))
        activity, confidence = self.classes[idx], float(decision_probs[idx])

        if apply_rules and accel_var < STILL_VAR_THRESHOLD and gyro_var < STILL_VAR_THRESHOLD:
            activity, confidence, rule = "Still", 1.0, "low-variance-still"
            if smoother:
                smoother.ema = np.eye(len(self.classes))[self.classes.index("Still")]
        else:
            if confidence < CONFIDENCE_THRESHOLD:
                activity = UNCERTAIN
            if apply_rules:
                total_var = accel_var + gyro_var
                if total_var < INTENSITY_SLIGHT:
                    allowed = ACTIVITIES_SLIGHT
                elif total_var < INTENSITY_STAIRS:
                    allowed = ACTIVITIES_STAIRS
                elif total_var < INTENSITY_WALK_JOG:
                    allowed = ACTIVITIES_WALK_JOG
                else:
                    allowed = None
                if allowed and activity not in allowed and activity != UNCERTAIN:
                    best = max((i for i, c in enumerate(self.classes) if c in allowed),
                               key=lambda i: decision_probs[i])
                    activity, confidence, rule = self.classes[best], float(decision_probs[best]), "intensity-gate"

        final = smoother.vote(activity) if smoother else activity
        return {
            "activity": final,
            "raw_activity": activity,
            "model_activity": model_activity,
            "model_confidence": round(float(probs[model_idx]), 4),
            "confidence": round(confidence, 4),
            "rule": rule,
            "all_probs": {c: round(float(p), 4) for c, p in zip(self.classes, probs)},
            "motion": {"accel_var": round(accel_var, 4), "gyro_var": round(gyro_var, 4)},
        }
