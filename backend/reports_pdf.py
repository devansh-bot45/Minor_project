"""Simple one-session PDF report (fpdf2, core fonts only)."""

from fpdf import FPDF

from storage import SECONDS_PER_PREDICTION


def _fmt_secs(s):
    s = int(round(s))
    return f"{s // 60}m {s % 60:02d}s"


def _latin1(text):
    return str(text).encode("latin-1", "replace").decode("latin-1")


def session_pdf(session):
    """`session` is Store.session_detail(). Returns PDF bytes."""
    preds = session["predictions"]
    summary = session["summary"]

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("helvetica", "B", 18)
    pdf.cell(0, 12, "Activity Session Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, "Human Activity Recognition - Conv1D + BiLSTM + Attention model", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.set_text_color(0, 0, 0)
    rows = [
        ("Session", session["name"]),
        ("Source", {"live": "Live phone sensors", "csv": "Uploaded CSV file",
                    "replay": "Replayed recording (not live)"}.get(session["source"], session["source"])),
        ("Started (UTC)", session["started_at"].replace("T", " ")),
        ("Recording time", _fmt_secs(session.get("active_seconds") or 0)),
        ("Predictions", f"{summary['predictions']} (one per {SECONDS_PER_PREDICTION:g} s window step)"),
        ("Most frequent activity", summary["dominant_activity"] or "-"),
    ]
    for k, v in rows:
        pdf.set_font("helvetica", "B", 11)
        pdf.cell(55, 8, k)
        pdf.set_font("helvetica", "", 11)
        pdf.cell(0, 8, _latin1(v), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # Activity breakdown with simple horizontal bars
    pdf.set_font("helvetica", "B", 13)
    pdf.cell(0, 10, "Activity breakdown (approximate time)", new_x="LMARGIN", new_y="NEXT")
    total = sum(summary["seconds_by_activity"].values()) or 1
    for act, secs in summary["seconds_by_activity"].items():
        pct = secs / total
        pdf.set_font("helvetica", "", 11)
        pdf.cell(40, 8, _latin1(act))
        x, y = pdf.get_x(), pdf.get_y() + 2
        pdf.set_fill_color(230, 233, 240)
        pdf.rect(x, y, 100, 4, style="F")
        pdf.set_fill_color(37, 99, 235)
        pdf.rect(x, y, max(100 * pct, 0.5), 4, style="F")
        pdf.set_x(x + 104)
        pdf.cell(0, 8, f"{_fmt_secs(secs)}  ({pct * 100:.0f}%)", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # Timeline table: merge consecutive identical activities into segments
    pdf.set_font("helvetica", "B", 13)
    pdf.cell(0, 10, "Timeline", new_x="LMARGIN", new_y="NEXT")
    segments = []
    for p in preds:
        if segments and segments[-1]["activity"] == p["activity"]:
            segments[-1]["end"] = p["t_sec"] + SECONDS_PER_PREDICTION
            segments[-1]["conf"].append(p["confidence"])
        else:
            segments.append({"activity": p["activity"], "start": p["t_sec"],
                             "end": p["t_sec"] + SECONDS_PER_PREDICTION, "conf": [p["confidence"]]})
    pdf.set_font("helvetica", "B", 10)
    for w, h in ((35, "From"), (35, "To"), (60, "Activity"), (40, "Avg. confidence")):
        pdf.cell(w, 7, h, border="B")
    pdf.ln()
    pdf.set_font("helvetica", "", 10)
    for seg in segments[:200]:
        pdf.cell(35, 6, _fmt_secs(seg["start"]))
        pdf.cell(35, 6, _fmt_secs(seg["end"]))
        pdf.cell(60, 6, _latin1(seg["activity"]))
        pdf.cell(40, 6, f"{sum(seg['conf']) / len(seg['conf']) * 100:.0f}%", new_x="LMARGIN", new_y="NEXT")
    if len(segments) > 200:
        pdf.cell(0, 6, f"... {len(segments) - 200} more segments (see CSV export)", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(6)
    pdf.set_font("helvetica", "I", 9)
    pdf.set_text_color(110, 110, 110)
    pdf.multi_cell(0, 5, "Predictions come from a machine-learning model and can be wrong. Times are estimated "
                         "from the number of prediction windows and are not medical measurements.")
    return bytes(pdf.output())
