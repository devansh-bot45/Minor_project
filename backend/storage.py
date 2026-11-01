"""
Local session/prediction storage using SQLite (Python standard library, no setup needed).
The database file defaults to backend/data/har.db and can be changed with HAR_DB_PATH.
"""

import json
import os
import sqlite3
import uuid
from collections import Counter
from datetime import datetime, timezone

from preprocessing import FS, STEP_SIZE

# Each stored prediction covers one window step (30 samples = 1.5 s of new data).
SECONDS_PER_PREDICTION = STEP_SIZE / FS

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id             TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    source         TEXT NOT NULL,      -- live | csv | replay
    started_at     TEXT NOT NULL,      -- ISO-8601 UTC
    ended_at       TEXT,
    active_seconds REAL DEFAULT 0,     -- recording time excluding pauses
    meta           TEXT                -- JSON (device info, file name, ...)
);
CREATE TABLE IF NOT EXISTS predictions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id     TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    t_sec          REAL NOT NULL,      -- seconds from session start (active time)
    created_at     TEXT NOT NULL,
    activity       TEXT NOT NULL,      -- final (smoothed) activity shown to the user
    model_activity TEXT NOT NULL,      -- pure model argmax
    confidence     REAL NOT NULL,
    rule           TEXT,               -- post-processing rule that changed the label, if any
    probs          TEXT NOT NULL       -- JSON {class: probability}
);
CREATE INDEX IF NOT EXISTS idx_predictions_session ON predictions(session_id, t_sec);
"""

VALID_SOURCES = {"live", "csv", "replay"}


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self):
        # A new connection per call keeps this safe with Flask's threaded dev server.
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    # ── Sessions ──────────────────────────────────────────────────────────────
    def create_session(self, name, source, meta=None):
        if source not in VALID_SOURCES:
            raise ValueError(f"source must be one of {sorted(VALID_SOURCES)}")
        sid = uuid.uuid4().hex[:12]
        name = (name or "").strip()[:80] or f"{source.title()} session"
        with self._conn() as c:
            c.execute("INSERT INTO sessions (id, name, source, started_at, meta) VALUES (?, ?, ?, ?, ?)",
                      (sid, name, source, _now(), json.dumps(meta or {})))
        return self.get_session(sid)

    def get_session(self, sid):
        with self._conn() as c:
            row = c.execute("SELECT * FROM sessions WHERE id = ?", (sid,)).fetchone()
        return self._session_dict(row) if row else None

    def end_session(self, sid, active_seconds=None):
        with self._conn() as c:
            if active_seconds is None:
                c.execute("UPDATE sessions SET ended_at = ? WHERE id = ?", (_now(), sid))
            else:
                c.execute("UPDATE sessions SET ended_at = ?, active_seconds = ? WHERE id = ?",
                          (_now(), float(active_seconds), sid))
        return self.get_session(sid)

    def delete_session(self, sid):
        with self._conn() as c:
            return c.execute("DELETE FROM sessions WHERE id = ?", (sid,)).rowcount > 0

    def list_sessions(self, limit=100):
        with self._conn() as c:
            rows = c.execute("SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?", (limit,)).fetchall()
            counts = c.execute(
                "SELECT session_id, activity, COUNT(*) AS n FROM predictions GROUP BY session_id, activity"
            ).fetchall()
        per_session = {}
        for r in counts:
            per_session.setdefault(r["session_id"], Counter())[r["activity"]] = r["n"]
        out = []
        for row in rows:
            s = self._session_dict(row)
            s["summary"] = self._summary(per_session.get(s["id"], Counter()))
            out.append(s)
        return out

    # ── Predictions ───────────────────────────────────────────────────────────
    def add_predictions(self, sid, items):
        """items: list of (t_sec, prediction dict from HARModel.postprocess)."""
        now = _now()
        with self._conn() as c:
            c.executemany(
                "INSERT INTO predictions (session_id, t_sec, created_at, activity, model_activity, "
                "confidence, rule, probs) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [(sid, float(t), now, p["activity"], p["model_activity"], float(p["confidence"]),
                  p.get("rule"), json.dumps(p["all_probs"])) for t, p in items],
            )
            if items:
                # Keep active_seconds up to date even if the session is never ended cleanly.
                c.execute("UPDATE sessions SET active_seconds = MAX(active_seconds, ?) WHERE id = ?",
                          (float(items[-1][0]) + SECONDS_PER_PREDICTION, sid))

    def get_predictions(self, sid):
        with self._conn() as c:
            rows = c.execute("SELECT * FROM predictions WHERE session_id = ? ORDER BY t_sec, id",
                             (sid,)).fetchall()
        return [{
            "t_sec": r["t_sec"], "created_at": r["created_at"], "activity": r["activity"],
            "model_activity": r["model_activity"], "confidence": r["confidence"],
            "rule": r["rule"], "all_probs": json.loads(r["probs"]),
        } for r in rows]

    def session_detail(self, sid):
        s = self.get_session(sid)
        if not s:
            return None
        preds = self.get_predictions(sid)
        s["predictions"] = preds
        s["summary"] = self._summary(Counter(p["activity"] for p in preds))
        return s

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _summary(counter):
        total = sum(counter.values())
        return {
            "predictions": total,
            "dominant_activity": counter.most_common(1)[0][0] if total else None,
            # Approximate: every prediction stands for one 1.5 s window step.
            "seconds_by_activity": {a: round(n * SECONDS_PER_PREDICTION, 1) for a, n in counter.most_common()},
        }

    @staticmethod
    def _session_dict(row):
        d = dict(row)
        d["meta"] = json.loads(d["meta"] or "{}")
        return d
