"""API tests with a fake model (fast, no TensorFlow)."""
import io

from conftest import still_window, walking_window


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json["status"] == "ok"
    assert "Walking" in r.json["classes"]


def test_model_info(client):
    r = client.get("/api/model").json
    assert r["input_shape"] == [60, 8]
    assert r["window_samples"] == 60 and r["sampling_rate_hz"] == 20


def test_predict_valid_window(client):
    r = client.post("/api/predict", json={"data": walking_window()})
    assert r.status_code == 200
    body = r.json
    assert body["activity"] == "Walking"
    assert body["model_activity"] == "Walking"
    assert abs(sum(body["all_probs"].values()) - 1) < 0.2


def test_legacy_predict_endpoint(client):
    r = client.post("/predict", json={"data": walking_window()})
    assert r.status_code == 200 and r.json["smoothed_activity"] == "Walking"


def test_still_rule_overrides_model(client):
    body = client.post("/api/predict", json={"data": still_window()}).json
    assert body["activity"] == "Still"
    assert body["model_activity"] == "Walking"   # the UI can show that a rule was applied
    assert body["rule"] == "low-variance-still"


def test_predict_rejects_bad_input(client):
    assert client.post("/api/predict", json={}).status_code == 400
    assert client.post("/api/predict", json={"data": [[1, 2, 3]] * 60}).status_code == 400
    assert client.post("/api/predict", json={"data": [[0] * 6] * 30}).status_code == 400
    assert client.post("/api/predict", json={"data": [["a"] * 6] * 60}).status_code == 400
    assert client.post("/api/predict", data="not json", content_type="text/plain").status_code == 400
    r = client.post("/api/predict", json={"data": [[float("nan")] * 6] * 60})
    assert r.status_code == 400


def test_session_flow_and_exports(client):
    s = client.post("/api/sessions", json={"name": "Test walk", "source": "live"})
    assert s.status_code == 201
    sid = s.json["id"]
    for i in range(4):
        r = client.post("/api/predict", json={"data": walking_window(i), "session_id": sid, "t_sec": i * 1.5})
        assert r.status_code == 200
    client.post(f"/api/sessions/{sid}/end", json={"active_seconds": 6})

    detail = client.get(f"/api/sessions/{sid}").json
    assert len(detail["predictions"]) == 4
    assert detail["summary"]["dominant_activity"] == "Walking"
    assert detail["active_seconds"] == 6

    listing = client.get("/api/sessions").json["sessions"]
    assert listing[0]["id"] == sid and listing[0]["summary"]["predictions"] == 4

    csv_resp = client.get(f"/api/sessions/{sid}/export.csv")
    assert csv_resp.status_code == 200
    lines = csv_resp.data.decode().strip().splitlines()
    assert lines[0].startswith("t_sec,activity") and len(lines) == 5

    pdf = client.get(f"/api/sessions/{sid}/report.pdf")
    assert pdf.status_code == 200 and pdf.data[:4] == b"%PDF"

    assert client.delete(f"/api/sessions/{sid}").status_code == 200
    assert client.get(f"/api/sessions/{sid}").status_code == 404


def test_unknown_session_rejected(client):
    r = client.post("/api/predict", json={"data": walking_window(), "session_id": "nope"})
    assert r.status_code == 404


def test_invalid_session_source(client):
    assert client.post("/api/sessions", json={"source": "fake"}).status_code == 400


def _wide_csv(rows, with_time=True):
    lines = ["timestamp_ms,ax,ay,az,gx,gy,gz" if with_time else "ax,ay,az,gx,gy,gz"]
    for i, r in enumerate(rows):
        vals = ",".join(f"{v:.4f}" for v in r)
        lines.append(f"{i * 50},{vals}" if with_time else vals)
    return "\n".join(lines).encode()


def test_csv_upload_wide_format_and_save(client):
    rows = walking_window(1) * 3  # 180 samples = 9 s -> 5 windows
    data = {"file": (io.BytesIO(_wide_csv(rows)), "walk.csv"), "save": "true"}
    r = client.post("/api/predict/csv", data=data, content_type="multipart/form-data")
    assert r.status_code == 200, r.json
    assert r.json["info"]["format"] == "wide"
    assert len(r.json["timeline"]) == 5
    assert r.json["session"]["source"] == "csv"
    assert client.get("/api/sessions").json["sessions"][0]["summary"]["predictions"] == 5


def test_csv_upload_android_format(client):
    lines = ["Timestamp,Sensor Type,Value1,Value2,Value3"]
    for i in range(200):
        t = 1_000_000 + i * 50
        lines.append(f"{t},1,0.1,0.2,9.8")
        lines.append(f"{t},4,0.0,0.0,0.0")
    data = {"file": (io.BytesIO("\n".join(lines).encode()), "still.csv")}
    r = client.post("/api/predict/csv", data=data, content_type="multipart/form-data")
    assert r.status_code == 200, r.json
    assert r.json["info"]["format"] == "android-logger"
    assert all(p["activity"] == "Still" for p in r.json["timeline"])


def test_csv_upload_errors(client):
    r = client.post("/api/predict/csv", data={}, content_type="multipart/form-data")
    assert r.status_code == 400
    bad = {"file": (io.BytesIO(b"foo,bar\n1,2\n"), "bad.csv")}
    r = client.post("/api/predict/csv", data=bad, content_type="multipart/form-data")
    assert r.status_code == 400 and "Unrecognised" in r.json["error"]
    short = {"file": (io.BytesIO(_wide_csv(walking_window()[:20])), "short.csv")}
    r = client.post("/api/predict/csv", data=short, content_type="multipart/form-data")
    assert r.status_code == 400 and "Not enough" in r.json["error"]


def test_csv_gap_is_not_interpolated(client):
    """A 10 s hole in the timestamps must split the data, not be filled with a flat line."""
    rows = walking_window(2) * 2
    lines = ["timestamp_ms,ax,ay,az,gx,gy,gz"]
    for i, r in enumerate(rows):
        t = i * 50 + (10_000 if i >= 60 else 0)
        lines.append(f"{t}," + ",".join(f"{v:.4f}" for v in r))
    data = {"file": (io.BytesIO("\n".join(lines).encode()), "gap.csv")}
    r = client.post("/api/predict/csv", data=data, content_type="multipart/form-data")
    assert r.status_code == 200
    assert r.json["info"]["segments"] == 2
    assert len(r.json["timeline"]) == 2   # one window per 60-sample segment


def test_csv_gap_with_epoch_timestamps(client):
    """Same as above with real-world millisecond epoch timestamps (~1.7e12)."""
    rows = walking_window(3) * 2
    lines = ["timestamp_ms,ax,ay,az,gx,gy,gz"]
    for i, r in enumerate(rows):
        t = 1_777_000_000_000 + i * 50 + (60_000 if i >= 60 else 0)
        lines.append(f"{t}," + ",".join(f"{v:.4f}" for v in r))
    data = {"file": (io.BytesIO("\n".join(lines).encode()), "gap.csv")}
    r = client.post("/api/predict/csv", data=data, content_type="multipart/form-data")
    assert r.status_code == 200
    assert r.json["info"]["segments"] == 2 and len(r.json["timeline"]) == 2


def test_samples_endpoint(client):
    samples = client.get("/api/samples").json["samples"]
    assert any(s["id"] == "still" for s in samples)
    r = client.get("/api/samples/still").json
    assert r["label"] == "Still" and len(r["samples"]) >= 60 and len(r["samples"][0]) == 6
    assert client.get("/api/samples/nope").status_code == 404
