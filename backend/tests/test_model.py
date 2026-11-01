"""Tests with the REAL Keras model and the project's own recordings (slower)."""
import os

import numpy as np
import pytest

from conftest import BACKEND, ROOT
from preprocessing import FRAME_SIZE, parse_sensor_csv, preprocess_window, windows_from_segments


@pytest.fixture(scope="module", params=[".", "model_v2"])
def har(request):
    """The original model and the retrained v2 model (skipped if model_v2/ is absent)."""
    from model_service import HARModel
    model_dir = os.path.join(BACKEND, request.param)
    if not os.path.exists(os.path.join(model_dir, "har_model.keras")):
        pytest.skip(f"{request.param} has no model")
    return HARModel(model_dir)


def test_artifacts_consistent(har):
    assert har.classes == ["Hand Activity", "Jogging", "Sports", "Stairs", "Still", "Walking"]
    assert har.input_shape == [60, 8]
    assert har.model.output_shape[-1] == 6


def test_preprocess_shape():
    out = preprocess_window(np.random.default_rng(0).normal(size=(FRAME_SIZE, 6)))
    assert out.shape == (60, 8)


def test_probabilities_are_valid(har):
    w = np.random.default_rng(1).normal(0, 2, size=(3, 60, 6))
    probs = har.predict_probs(w)
    assert probs.shape == (3, 6)
    assert np.allclose(probs.sum(axis=1), 1, atol=1e-5)


def _majority(har, filename):
    path = os.path.join(ROOT, filename)
    if not os.path.exists(path):
        pytest.skip(f"{filename} not present")
    with open(path, "rb") as f:
        segments, _ = parse_sensor_csv(f.read())
    _, windows = windows_from_segments(segments)
    preds = [har.classes[i] for i in har.predict_probs(windows).argmax(axis=1)]
    return max(set(preds), key=preds.count)


def test_still_recording_predicts_still(har):
    assert _majority(har, "still data.csv") == "Still"


def test_jogging_recording_predicts_jogging(har):
    assert _majority(har, "jogging.csv") == "Jogging"


def test_recording_flags_follow_the_model_directory(tmp_path):
    import json
    import recordings
    default = {r["id"]: r["used_in_training"] for r in recordings.available()}
    assert default["walking2"] is True and default["stairs1"] is False
    (tmp_path / "trained_on_recordings.json").write_text(json.dumps({"used_in_training": {"walking2": False}}))
    flags = {r["id"]: r["used_in_training"] for r in recordings.available(str(tmp_path))}
    assert flags["walking2"] is False and flags["still"] is True and flags["eating"] is None
