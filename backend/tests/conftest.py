import os
import sys

import numpy as np
import pytest

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BACKEND)
sys.path.insert(0, BACKEND)

from app import create_app  # noqa: E402
from model_service import HARModel  # noqa: E402
from storage import Store  # noqa: E402

CLASSES = ["Hand Activity", "Jogging", "Sports", "Stairs", "Still", "Walking"]


class FakeModel(HARModel):
    """Skips TensorFlow; returns 'Walking' for moving windows. Rules/smoothing are the real ones."""

    def __init__(self):
        self.classes = CLASSES
        self.input_shape = [60, 8]
        self.param_count = 0

    def predict_probs(self, raw_windows):
        out = np.full((len(raw_windows), len(CLASSES)), 0.02)
        out[:, CLASSES.index("Walking")] = 0.9
        return out


def walking_window(seed=0):
    """Synthetic 60×6 window with strong periodic motion (only for API tests)."""
    rng = np.random.default_rng(seed)
    t = np.arange(60) / 20.0
    acc = np.column_stack([3 * np.sin(2 * np.pi * 2 * t), 9.8 + 4 * np.sin(2 * np.pi * 2 * t + 1),
                           2 * np.cos(2 * np.pi * 2 * t)])
    gyro = np.column_stack([1.5 * np.sin(2 * np.pi * 2 * t), np.cos(2 * np.pi * 2 * t), 0.5 * np.sin(t)])
    return (np.hstack([acc, gyro]) + rng.normal(0, 0.05, (60, 6))).tolist()


def still_window():
    return [[0.0, 0.0, 9.81, 0.0, 0.0, 0.0]] * 60


@pytest.fixture
def client(tmp_path):
    app = create_app(model=FakeModel(), store=Store(str(tmp_path / "test.db")))
    app.testing = True
    return app.test_client()
