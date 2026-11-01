"""
Train optimized Hybrid CNN + BiLSTM + Attention model for HAR.
Run from Minor_HAR folder: python backend/train_model.py
Saves: backend/har_model.keras, scaler.pkl, label_encoder.pkl,
       activity_names.pkl, confusion_matrix.png, report.txt
"""

import os
import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import keras
from keras.models import Model
from keras.layers import (
    Input, Conv1D, MaxPooling1D, Dense, Dropout,
    BatchNormalization, LSTM, Bidirectional,
    GlobalAveragePooling1D, Multiply, Permute,
    RepeatVector, Flatten, Activation, Lambda,
    Add, Concatenate
)
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

try:
    import keras.ops as ops
except ImportError:
    import keras.backend as ops

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR      = "backend"
FREQUENCY       = 20        # Hz
TIME_PERIOD     = 3          # seconds
FRAME_SIZE      = FREQUENCY * TIME_PERIOD   # 60 samples per window
NUM_CHANNELS    = 8                         # accel(X,Y,Z) + gyro(X,Y,Z) + mags
EPOCHS          = 100
BATCH_SIZE      = 64
TEST_SIZE       = 0.2
RANDOM_STATE    = 42

ACTIVITY_NAMES_SORTED = sorted(["Walking", "Jogging", "Stairs", "Still", "Hand Activity", "Sports"])
NUM_CLASSES = len(ACTIVITY_NAMES_SORTED)  # 6


# ── Focal Loss for class imbalance ────────────────────────────────────────────
def focal_loss(gamma=2.0, alpha=0.25):
    """Focal loss — down-weights easy/majority samples, focuses on hard ones."""
    def focal_loss_fn(y_true, y_pred):
        y_true = ops.cast(y_true, 'int32')
        y_true_one_hot = ops.one_hot(ops.reshape(y_true, [-1]), NUM_CLASSES)
        y_pred = ops.clip(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = -y_true_one_hot * ops.log(y_pred)
        weight = alpha * y_true_one_hot * ops.power(1.0 - y_pred, gamma)
        loss = weight * cross_entropy
        return ops.sum(loss, axis=-1)
    return focal_loss_fn


# ── Label Smoothing ───────────────────────────────────────────────────────────
def smooth_labels(y, num_classes, smoothing=0.1):
    """Apply label smoothing to prevent overconfident predictions."""
    one_hot = np.eye(num_classes)[y]
    one_hot = one_hot * (1.0 - smoothing) + smoothing / num_classes
    return one_hot


def preprocess_numpy(X_file, y_file):
    print("=" * 60)
    print("  LOADING PREPARED ARRAYS")
    print("=" * 60)

    X_all = np.load(X_file)
    y_raw = np.load(y_file)

    label_encoder = LabelEncoder()
    label_encoder.fit(ACTIVITY_NAMES_SORTED)
    y_all = label_encoder.transform(y_raw)

    print("  Fitting global StandardScaler on 8 channels...")
    N, steps, channels = X_all.shape
    X_flat = X_all.reshape(-1, channels)

    scaler = StandardScaler()
    X_flat_scaled = scaler.fit_transform(X_flat)
    X_all_scaled = X_flat_scaled.reshape(N, steps, channels)

    print("\n  Class distribution:")
    unique, counts = np.unique(y_all, return_counts=True)
    for u, c in zip(unique, counts):
        name = label_encoder.inverse_transform([u])[0]
        print(f"    {name:15s}: {c:5d} frames ({c/len(y_all)*100:.1f}%)")

    return X_all_scaled, y_all, label_encoder, scaler


def attention_block(inputs):
    """Temporal attention: learn which timesteps matter most."""
    features = inputs.shape[-1]

    # Compute attention scores
    a = Dense(features, activation='tanh')(inputs)
    a = Dense(1, activation='linear')(a)
    a = Flatten()(a)
    a = Activation('softmax')(a)
    a = RepeatVector(features)(a)
    a = Permute((2, 1))(a)

    # Apply attention
    output = Multiply()([inputs, a])
    return output


def create_model(input_shape, num_classes):
    """
    Optimized Hybrid Conv1D + Bidirectional LSTM + Attention model.
    - Deeper CNN feature extraction (3 blocks)
    - Bidirectional LSTM for better temporal learning
    - Temporal attention to focus on discriminative timesteps
    - Reduced regularization to prevent underfitting
    """
    inputs = Input(shape=input_shape)

    # Feature Extraction Block 1
    x = Conv1D(64, kernel_size=5, activation="relu", padding="same",
               kernel_regularizer=l2(5e-5))(inputs)
    x = BatchNormalization()(x)
    x = Conv1D(64, kernel_size=3, activation="relu", padding="same",
               kernel_regularizer=l2(5e-5))(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.15)(x)

    # Feature Extraction Block 2
    x = Conv1D(128, kernel_size=3, activation="relu", padding="same",
               kernel_regularizer=l2(5e-5))(x)
    x = BatchNormalization()(x)
    x = Conv1D(128, kernel_size=3, activation="relu", padding="same",
               kernel_regularizer=l2(5e-5))(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)

    # Feature Extraction Block 3 (NEW — deeper feature extraction)
    x = Conv1D(256, kernel_size=3, activation="relu", padding="same",
               kernel_regularizer=l2(5e-5))(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)

    # Temporal Learning: Bidirectional LSTM (upgraded from unidirectional)
    x = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(5e-5)))(x)
    x = Dropout(0.25)(x)

    # Temporal Attention
    x = attention_block(x)

    x = Bidirectional(LSTM(64, return_sequences=False, kernel_regularizer=l2(5e-5)))(x)
    x = Dropout(0.3)(x)

    # Classifier (reduced dropout)
    x = Dense(128, activation="relu", kernel_regularizer=l2(5e-5))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu", kernel_regularizer=l2(5e-5))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=["accuracy"],
    )
    return model


def evaluate_model(model, X_test, y_test, label_encoder, output_dir):
    print("\n" + "=" * 60)
    print("  EVALUATING MODEL")
    print("=" * 60)

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n  Test Accuracy: {acc * 100:.2f}%\n")

    target_names = list(label_encoder.classes_)

    report = classification_report(
        y_test, y_pred,
        labels=range(len(target_names)),
        target_names=target_names,
        zero_division=0,
    )
    print("  Classification Report:")
    print("  " + "-" * 56)
    for line in report.split("\n"):
        print(f"  {line}")

    report_path = os.path.join(output_dir, "report.txt")
    with open(report_path, "w") as f:
        f.write(f"Test Accuracy: {acc * 100:.2f}%\n\n")
        f.write("Classification Report\n")
        f.write("=" * 56 + "\n")
        f.write(report)
    print(f"\n  Report saved -> {report_path}")

    cm = confusion_matrix(y_test, y_pred, labels=range(len(target_names)))
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.title(f"Confusion Matrix — Hybrid BiLSTM+Attention (Accuracy: {acc*100:.1f}%)", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved -> {cm_path}")

    return acc


def save_artifacts(model, scaler, label_encoder, output_dir):
    print("\n" + "=" * 60)
    print("  SAVING ARTIFACTS")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "har_model.keras")
    model.save(model_path)
    print(f"  Model saved          -> {model_path}")

    scaler_path = os.path.join(output_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Scaler saved         -> {scaler_path}")

    encoder_path = os.path.join(output_dir, "label_encoder.pkl")
    with open(encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)
    print(f"  Label encoder saved  -> {encoder_path}")

    activity_names = {}
    for i, name in enumerate(label_encoder.classes_):
        activity_names[i] = name

    names_path = os.path.join(output_dir, "activity_names.pkl")
    with open(names_path, "wb") as f:
        pickle.dump(activity_names, f)
    print(f"  Activity names saved -> {names_path}")
    print(f"  Classes: {activity_names}")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(BASE_DIR)

    x_path = os.path.join(ROOT_DIR, "X_all.npy")
    y_path = os.path.join(ROOT_DIR, "y_all.npy")

    X_all, y_all, label_encoder, scaler = preprocess_numpy(x_path, y_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_all,
    )
    print(f"\n  Train set: {X_train.shape}  |  Test set: {X_test.shape}")

    # Compute class weights for balanced training
    from sklearn.utils.class_weight import compute_class_weight
    class_weights_arr = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = dict(zip(np.unique(y_train), class_weights_arr))
    print("\n  Class weights:")
    for idx, w in class_weights.items():
        name = label_encoder.inverse_transform([idx])[0]
        print(f"    {name:15s}: {w:.3f}")

    print("\n" + "=" * 60)
    print("  TRAINING OPTIMIZED MODEL (BiLSTM + Attention + Focal Loss)")
    print("=" * 60)
    model = create_model(
        input_shape=(FRAME_SIZE, NUM_CHANNELS),
        num_classes=NUM_CLASSES
    )
    model.summary()

    early_stop = EarlyStopping(
        monitor="val_accuracy",
        patience=15,
        restore_best_weights=True,
        verbose=1,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1,
    )

    # Shuffle training data
    shuffle_idx = np.random.permutation(len(X_train))
    X_train = X_train[shuffle_idx]
    y_train = y_train[shuffle_idx]

    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.2,
        callbacks=[early_stop, reduce_lr],
        class_weight=class_weights,
        shuffle=True,
        verbose=1,
    )

    evaluate_model(model, X_test, y_test, label_encoder, OUTPUT_DIR)
    save_artifacts(model, scaler, label_encoder, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("  ALL DONE!")
    print(f"  Model: {NUM_CLASSES} classes, {NUM_CHANNELS}-channel input")
    print("  Architecture: Conv1D(3 blocks) + BiLSTM + Attention + Focal Loss")
    print("  Run the app:  python backend/app.py")
    print("=" * 60 + "\n")