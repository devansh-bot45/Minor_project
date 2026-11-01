"""
Flask backend for Human Activity Recognition.
Run from the backend folder:  python app.py

Core features (no credentials needed):
  - live window prediction with per-session smoothing
  - CSV upload prediction
  - session history in SQLite + CSV/PDF export
Optional: Google sign-in / MongoDB / Gemini reports (see cloud_reports.py).
"""

import csv
import io
import json
import os
import threading

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request
from flask_cors import CORS

from preprocessing import FRAME_SIZE, FS, STEP_SIZE, parse_sensor_csv, validate_window, windows_from_segments
from model_service import (
    CONFIDENCE_THRESHOLD, EMA_ALPHA, SMOOTHING_WINDOW, TEMPERATURE, SessionSmoother,
)
from storage import SECONDS_PER_PREDICTION, Store
import recordings

BASE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE, ".env"))

MAX_UPLOAD_MB = 20

# Which trained model to serve. Default: the retrained model in backend/model_v2 if present.
# HAR_MODEL_DIR=. selects the original model (backend/har_model.keras).
MODEL_DIR = os.path.normpath(os.path.join(
    BASE, os.getenv("HAR_MODEL_DIR") or ("model_v2" if os.path.exists(os.path.join(BASE, "model_v2", "har_model.keras")) else ".")))
MODEL_VERSION = "original" if MODEL_DIR == BASE else os.path.relpath(MODEL_DIR, BASE)


def create_app(model=None, store=None):
    """
    App factory. `model` and `store` can be injected (tests use a fake model);
    otherwise the real Keras model and the SQLite store are loaded.
    """
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
    CORS(app)  # the frontend normally uses the Vite proxy, but direct calls also work

    if model is None:
        from model_service import HARModel
        print("Loading model...")
        model = HARModel(MODEL_DIR)
        print(f"  [OK] Model loaded from {MODEL_DIR}: {model.classes}")
    if store is None:
        store = Store(os.getenv("HAR_DB_PATH", os.path.join(BASE, "data", "har.db")))

    # Smoothing state per live session (in memory; a restart simply starts fresh).
    smoothers = {}
    smoothers_lock = threading.Lock()

    cloud_enabled = False
    log_activity = None
    try:
        import cloud_reports
        cloud_enabled = cloud_reports.init_cloud(app)
        log_activity = cloud_reports.log_activity if cloud_enabled else None
    except ImportError:
        pass  # optional packages not installed
    print(f"  Cloud reports (Google/MongoDB/Gemini): {'enabled' if cloud_enabled else 'disabled'}")

    def error(msg, code=400):
        return jsonify({"status": "error", "error": msg}), code

    def load_evaluation():
        path = os.path.join(MODEL_DIR, "evaluation_report.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    # ── Info ──────────────────────────────────────────────────────────────────
    @app.route("/", methods=["GET"])
    @app.route("/api/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "ok",
            "model_loaded": True,
            "classes": model.classes,
            "storage": "sqlite",
            "model_version": MODEL_VERSION,
            "cloud_reports_enabled": cloud_enabled,
        })

    @app.route("/api/model", methods=["GET"])
    def model_info():
        return jsonify({
            "model_version": MODEL_VERSION,
            "architecture": "Conv1D (3 blocks) + BiLSTM + Temporal Attention + BiLSTM + Dense",
            "classes": model.classes,
            "input_shape": model.input_shape,
            "parameters": model.param_count,
            "sampling_rate_hz": FS,
            "window_samples": FRAME_SIZE,
            "window_step_samples": STEP_SIZE,
            "raw_channels": ["ax", "ay", "az", "gx", "gy", "gz"],
            "units": {"accelerometer": "m/s² (including gravity)", "gyroscope": "rad/s"},
            "postprocessing": {
                "confidence_threshold": CONFIDENCE_THRESHOLD,
                "temperature": TEMPERATURE,
                "ema_alpha": EMA_ALPHA,
                "majority_vote_window": SMOOTHING_WINDOW,
            },
            "evaluation": load_evaluation(),
        })

    # ── Live prediction ───────────────────────────────────────────────────────
    @app.route("/predict", methods=["POST"])       # original endpoint, kept for compatibility
    @app.route("/api/predict", methods=["POST"])
    def predict():
        body = request.get_json(silent=True)
        if not isinstance(body, dict) or "data" not in body:
            return error("Request body must be JSON with a 'data' field (60 × 6 numbers).")
        try:
            window = validate_window(body["data"])
        except ValueError as e:
            return error(str(e))

        session_id = body.get("session_id")
        smoother = None
        if session_id:
            if not store.get_session(session_id):
                return error("Unknown session_id. Start a session first.", 404)
            with smoothers_lock:
                smoother = smoothers.setdefault(session_id, SessionSmoother())

        probs = model.predict_probs(window[None, ...])[0]
        with smoothers_lock:  # protects the session's smoothing state
            result = model.postprocess(probs, window, smoother)

        if session_id:
            t_sec = body.get("t_sec")
            t_sec = float(t_sec) if isinstance(t_sec, (int, float)) else 0.0
            store.add_predictions(session_id, [(t_sec, result)])
            result["t_sec"] = t_sec
        if log_activity:
            log_activity(request.headers.get("Authorization"), result["activity"], result["confidence"])

        result["smoothed_activity"] = result["activity"]  # field name used by the old frontend
        result["status"] = "ok"
        return jsonify(result)

    # ── CSV upload ────────────────────────────────────────────────────────────
    @app.route("/api/predict/csv", methods=["POST"])
    def predict_csv():
        upload = request.files.get("file")
        if upload is None or not upload.filename:
            return error("Attach a CSV file in the 'file' form field.")
        try:
            segments, info = parse_sensor_csv(upload.read())
        except ValueError as e:
            return error(str(e))
        times, windows = windows_from_segments(segments)
        if len(windows) == 0:
            return error(f"Not enough continuous data: need at least {FRAME_SIZE} samples "
                         f"({FRAME_SIZE / FS:.0f} s at {FS:.0f} Hz), got {info['samples']}.")

        probs = model.predict_probs(windows)
        smoother = SessionSmoother()  # treat the file like one continuous recording
        timeline = []
        for t, p, w in zip(times, probs, windows):
            r = model.postprocess(p, w, smoother)
            r["t_sec"] = round(float(t), 2)
            timeline.append(r)

        session = None
        if request.form.get("save", "").lower() in ("1", "true", "yes"):
            session = store.create_session(request.form.get("name") or upload.filename, "csv",
                                           {"file": upload.filename, **info})
            store.add_predictions(session["id"], [(r["t_sec"], r) for r in timeline])
            session = store.end_session(session["id"], info["duration_sec"])

        counts = {}
        for r in timeline:
            counts[r["activity"]] = counts.get(r["activity"], 0) + 1
        return jsonify({
            "status": "ok",
            "file": upload.filename,
            "info": info,
            "timeline": timeline,
            "summary": {
                "predictions": len(timeline),
                "seconds_by_activity": {a: round(n * SECONDS_PER_PREDICTION, 1) for a, n in
                                        sorted(counts.items(), key=lambda kv: -kv[1])},
            },
            "session": session,
        })

    # ── Recorded samples (demo replay) ────────────────────────────────────────
    @app.route("/api/samples", methods=["GET"])
    def list_samples():
        return jsonify({"samples": recordings.available(MODEL_DIR)})

    @app.route("/api/samples/<sample_id>", methods=["GET"])
    def get_sample(sample_id):
        rec = next((r for r in recordings.available(MODEL_DIR) if r["id"] == sample_id), None)
        if not rec:
            return error("Sample not found", 404)
        with open(recordings.path_for(rec), "rb") as f:
            segments, info = parse_sensor_csv(f.read())
        _, longest = max(segments, key=lambda s: len(s[1]))  # one continuous stretch, no gaps
        return jsonify({**rec, "info": info, "samples": [[round(float(v), 4) for v in row] for row in longest]})

    # ── Sessions / history ────────────────────────────────────────────────────
    @app.route("/api/sessions", methods=["POST"])
    def create_session():
        body = request.get_json(silent=True) or {}
        try:
            s = store.create_session(body.get("name"), body.get("source", "live"), body.get("meta"))
        except ValueError as e:
            return error(str(e))
        return jsonify(s), 201

    @app.route("/api/sessions", methods=["GET"])
    def list_sessions():
        return jsonify({"sessions": store.list_sessions()})

    @app.route("/api/sessions/<sid>", methods=["GET"])
    def get_session(sid):
        s = store.session_detail(sid)
        return jsonify(s) if s else error("Session not found", 404)

    @app.route("/api/sessions/<sid>/end", methods=["POST"])
    def end_session(sid):
        if not store.get_session(sid):
            return error("Session not found", 404)
        body = request.get_json(silent=True) or {}
        active = body.get("active_seconds")
        with smoothers_lock:
            smoothers.pop(sid, None)
        return jsonify(store.end_session(sid, float(active) if isinstance(active, (int, float)) else None))

    @app.route("/api/sessions/<sid>", methods=["DELETE"])
    def delete_session(sid):
        with smoothers_lock:
            smoothers.pop(sid, None)
        return jsonify({"deleted": True}) if store.delete_session(sid) else error("Session not found", 404)

    @app.route("/api/sessions/<sid>/export.csv", methods=["GET"])
    def export_csv(sid):
        s = store.session_detail(sid)
        if not s:
            return error("Session not found", 404)
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["t_sec", "activity", "model_activity", "confidence", "rule"] + model.classes)
        for p in s["predictions"]:
            writer.writerow([p["t_sec"], p["activity"], p["model_activity"], round(p["confidence"], 4),
                             p["rule"] or ""] + [p["all_probs"].get(c, "") for c in model.classes])
        return Response(buf.getvalue(), mimetype="text/csv", headers={
            "Content-Disposition": f'attachment; filename="har_session_{sid}.csv"'})

    @app.route("/api/sessions/<sid>/report.pdf", methods=["GET"])
    def export_pdf(sid):
        s = store.session_detail(sid)
        if not s:
            return error("Session not found", 404)
        try:
            from reports_pdf import session_pdf
        except ImportError:
            return error("PDF export needs the fpdf2 package (pip install fpdf2).", 501)
        return Response(session_pdf(s), mimetype="application/pdf", headers={
            "Content-Disposition": f'attachment; filename="har_session_{sid}.pdf"'})

    @app.errorhandler(413)
    def too_large(_):
        return error(f"File too large (max {MAX_UPLOAD_MB} MB).", 413)

    return app


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))  # 5000 is taken by AirPlay Receiver on macOS
    debug_mode = os.getenv("DEBUG", "false").lower() == "true"
    application = create_app()
    print("\n" + "=" * 50)
    print(f"  HAR Backend running at http://localhost:{port}")
    print("=" * 50 + "\n")
    # use_reloader=False: the reloader would load the TensorFlow model twice.
    application.run(debug=debug_mode, host="0.0.0.0", port=port, use_reloader=False)
