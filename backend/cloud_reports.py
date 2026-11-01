"""
OPTIONAL cloud features from the original project:
Google sign-in -> JWT, MongoDB activity logs, Gemini AI daily reports, report sharing and PDF.

Enabled only when MONGODB_URI, JWT_SECRET and GOOGLE_CLIENT_ID (or VITE_GOOGLE_CLIENT_ID)
are set and the optional packages are installed (pip install -r requirements-cloud.txt).
The core app (prediction, sessions, history, exports) never depends on this module.
"""

import collections
import io
import os
from datetime import datetime, timedelta
from functools import wraps

import jwt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bson.objectid import ObjectId
from flask import Blueprint, jsonify, request, send_file
from fpdf import FPDF
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token
from pymongo import MongoClient

from storage import SECONDS_PER_PREDICTION

try:
    import google.generativeai as genai
except ImportError:  # reports can still be listed/exported without Gemini
    genai = None

bp = Blueprint("cloud", __name__)

# MET values for the calorie estimate (MET * weight_kg * hours)
MET_VALUES = {
    "Still": 1.3, "Walking": 3.5, "Jogging": 7.5, "Stairs": 8.0,
    "Hand Activity": 2.0, "Sports": 6.0, "Uncertain": 1.0,
}

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID") or os.getenv("VITE_GOOGLE_CLIENT_ID")
JWT_SECRET = os.getenv("JWT_SECRET")
db = None


def init_cloud(app):
    """Register the blueprint. Returns True if cloud features are active."""
    global db
    if not (os.getenv("MONGODB_URI") and JWT_SECRET and GOOGLE_CLIENT_ID):
        return False
    client = MongoClient(os.getenv("MONGODB_URI"), serverSelectionTimeoutMS=3000)
    db = client.har_database
    if os.getenv("GEMINI_API_KEY") and genai:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    app.register_blueprint(bp)
    return True


def log_activity(auth_header, activity, confidence):
    """Store a prediction in MongoDB for a signed-in user (best effort, never raises)."""
    if db is None or not auth_header or activity == "Uncertain":
        return
    try:
        token = auth_header.split(" ")[1]
        data = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        user = db.users.find_one({"email": data["email"]})
        if user:
            db.activity_logs.insert_one({
                "user_id": user["_id"], "timestamp": datetime.utcnow(),
                "activity": activity, "confidence": float(confidence),
            })
    except Exception:
        pass


@bp.route("/api/auth/config", methods=["GET"])
def auth_config():
    return jsonify({"google_client_id": GOOGLE_CLIENT_ID})


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("Authorization")
        if not token:
            return jsonify({"error": "Token is missing"}), 401
        try:
            token = token.split(" ")[1] # Bearer token
            data = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
            current_user = db.users.find_one({"email": data["email"]})
            if not current_user:
                raise Exception("User not found")
        except Exception:
            return jsonify({"error": "Token is invalid"}), 401
        return f(current_user, *args, **kwargs)
    return decorated

@bp.route("/api/auth/google", methods=["POST"])
def google_auth():
    token = request.json.get("token")
    if not GOOGLE_CLIENT_ID:
        return jsonify({"error": "Server missing Google Client ID"}), 500
        
    try:
        if "access_token" in request.json:
            import requests
            access_token = request.json.get("access_token")
            user_info_url = f"https://www.googleapis.com/oauth2/v3/userinfo?access_token={access_token}"
            resp = requests.get(user_info_url)
            if resp.status_code != 200:
                raise Exception("Invalid access token")
            user_info = resp.json()
            email = user_info["email"]
            name = user_info.get("name", "")
        else:
            print(f"Verifying token for Client ID: {GOOGLE_CLIENT_ID[:10]}...")
            idinfo = id_token.verify_oauth2_token(token, google_requests.Request(), GOOGLE_CLIENT_ID)
            email = idinfo["email"]
            name = idinfo.get("name", "")
        
        user = db.users.find_one({"email": email})
        if not user:
            user = {"email": email, "name": name, "created_at": datetime.utcnow()}
            db.users.insert_one(user)
            
        jwt_token = jwt.encode(
            {"email": email, "exp": datetime.utcnow() + timedelta(days=7)}, 
            JWT_SECRET, 
            algorithm="HS256"
        )
        return jsonify({"token": jwt_token, "user": {"email": email, "name": name}})
    except Exception as e:
        print(f"Google Auth Error: {str(e)}")
        return jsonify({"error": str(e), "status": "unauthorized"}), 401


@bp.route("/api/reports/generate", methods=["POST"])
@token_required
def generate_report(current_user):
    if not os.getenv("GEMINI_API_KEY") or genai is None:
        return jsonify({"error": "Gemini API key not configured"}), 500
        
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    logs = list(db.activity_logs.find({"user_id": current_user["_id"], "timestamp": {"$gte": today}}))
    
    if not logs:
        return jsonify({"error": "No activity logged today"}), 400
        
    hourly_stats = {hour: collections.Counter() for hour in range(24)}
    total_counts = collections.Counter()
    
    for log in logs:
        act = log["activity"]
        # Ensure timestamp is a datetime object
        ts = log["timestamp"]
        if isinstance(ts, datetime):
            hour = ts.hour
            hourly_stats[hour][act] += 1
            total_counts[act] += 1
            
    # Convert prediction counts to minutes (one prediction per 1.5 s window step)
    stats = {
        "totals": {act: round((count * SECONDS_PER_PREDICTION) / 60, 2) for act, count in total_counts.items()},
        "hourly": {}
    }
    for hour in range(24):
        # Only include activities with > 0 minutes for efficiency
        hour_data = {act: round((count * SECONDS_PER_PREDICTION) / 60, 2) for act, count in hourly_stats[hour].items() if count > 0}
        stats["hourly"][str(hour)] = hour_data
        
    total_calories = 0
    for act, mins in stats["totals"].items():
        met = MET_VALUES.get(act, 1.0)
        total_calories += met * 70 * (mins / 60.0) # Assuming 70kg average
    
    stats["total_calories"] = round(total_calories, 1)

    summary_text = f"User {current_user['name']} had the following activities today: "
    for act, mins in stats["totals"].items():
        summary_text += f"{act}: {mins:.1f} minutes, "
    summary_text += f"Total estimated calories burned: {stats['total_calories']} kcal."
        
    prompt = (
        f"You are an expert AI health and fitness assistant. Based on the following raw sensor data summary for today, "
        f"write a comprehensive, highly professional, and encouraging daily health report for the user. "
        f"Your report must include:\n"
        f"1. A clear summary of the physical activities the user has engaged in throughout the day.\n"
        f"2. Insights into their movement patterns, highlighting any positive trends or areas for improvement.\n"
        f"3. A mention of their calorie burn ({stats['total_calories']} kcal) and how it relates to their activity levels.\n"
        f"4. Constructive, actionable suggestions for better health, posture, or maintaining an active lifestyle.\n"
        f"5. Any other relevant wellness advice based on the data provided.\n"
        f"Keep the tone supportive and clinical. DO NOT use any markdown characters (like **, ###, or *). Use plain text, uppercase words for headings, and standard dashes (-) for bullets to ensure the text is clean and professional.\n"
        f"Data: {summary_text}"
    )
    
    try:
        gen_model = genai.GenerativeModel("gemini-flash-latest")
        response = gen_model.generate_content(prompt)
        
        report = {
            "user_id": current_user["_id"],
            "date": today,
            "report_text": response.text,
            "stats": stats,
            "shared_with": []
        }
        
        existing = db.reports.find_one({"user_id": current_user["_id"], "date": today})
        if existing:
            db.reports.update_one({"_id": existing["_id"]}, {"$set": {"report_text": response.text, "stats": stats}})
        else:
            db.reports.insert_one(report)
            
        return jsonify({"message": "Report generated", "report": response.text})
    except Exception as e:
        return jsonify({"error": f"Failed to generate report: {str(e)}"}), 500


@bp.route("/api/reports", methods=["GET"])
@token_required
def get_reports(current_user):
    own_reports = list(db.reports.find({"user_id": current_user["_id"]}).sort("date", -1))
    shared_reports = list(db.reports.find({"shared_with": current_user["email"]}).sort("date", -1))
    
    for r in shared_reports:
        sender = db.users.find_one({"_id": r["user_id"]})
        if sender:
            r["sender_name"] = sender.get("name", "Unknown User")
            r["sender_email"] = sender.get("email", "")
        else:
            r["sender_name"] = "Unknown User"
            r["sender_email"] = ""
            
    for r in own_reports + shared_reports:
        r["_id"] = str(r["_id"])
        r["user_id"] = str(r["user_id"])
        r["date"] = r["date"].strftime("%Y-%m-%d")
        
    return jsonify({"own": own_reports, "shared": shared_reports})


@bp.route("/api/reports/share", methods=["POST"])
@token_required
def share_report(current_user):
    report_id = request.json.get("report_id")
    target_email = request.json.get("email")
    
    target_user = db.users.find_one({"email": target_email})
    if not target_user:
        return jsonify({"error": "User not found"}), 404
        
    result = db.reports.update_one(
        {"_id": ObjectId(report_id), "user_id": current_user["_id"]},
        {"$addToSet": {"shared_with": target_email}}
    )
    
    if result.modified_count == 0:
        return jsonify({"error": "Report not found or already shared"}), 400
        
    return jsonify({"message": f"Report shared with {target_email}"})

@bp.route("/api/reports/<report_id>/pdf", methods=["GET"])
@token_required
def generate_pdf_report(current_user, report_id):
    try:
        report = db.reports.find_one({"_id": ObjectId(report_id)})
        if not report:
            return jsonify({"error": "Report not found"}), 404
            
        if str(report["user_id"]) != str(current_user["_id"]) and current_user["email"] not in report.get("shared_with", []):
            return jsonify({"error": "Unauthorized"}), 403
            
        owner = db.users.find_one({"_id": report["user_id"]})
        owner_name = owner["name"] if owner else "Unknown User"
        
        class PDF(FPDF):
            def header(self):
                self.set_font("helvetica", "B", 20)
                self.set_text_color(41, 128, 185)
                self.cell(0, 15, "Official Health & Activity Report", align="C", new_x="LMARGIN", new_y="NEXT")
                self.set_line_width(0.5)
                self.set_draw_color(41, 128, 185)
                self.line(10, 25, 200, 25)
                self.ln(10)
                
            def footer(self):
                self.set_y(-15)
                self.set_font("helvetica", "I", 8)
                self.set_text_color(128, 128, 128)
                self.cell(0, 10, f"Page {self.page_no()}", align="C")

        pdf = PDF()
        pdf.add_page()
        
        pdf.set_font("helvetica", "B", 12)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(50, 10, "Patient Name:")
        pdf.set_font("helvetica", "", 12)
        pdf.cell(0, 10, owner_name, new_x="LMARGIN", new_y="NEXT")
        
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(40, 10, "Date of Report:")
        pdf.set_font("helvetica", "", 12)
        date_str = report["date"].strftime("%B %d, %Y") if isinstance(report["date"], datetime) else str(report["date"])
        pdf.cell(0, 10, date_str, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)
        
        # Summary Widgets
        calories = report.get("stats", {}).get("total_calories", 0)
        active_time = sum(report.get("stats", {}).get("totals", {}).values()) if report.get("stats") else 0
        intensity = "High" if calories > 450 else ("Moderate" if calories >= 240 else "Low")
        
        pdf.set_font("helvetica", "B", 11)
        pdf.set_fill_color(245, 247, 250)
        pdf.cell(60, 12, f"Energy: {calories} kcal", border=1, align="C", fill=True)
        pdf.cell(60, 12, f"Active Time: {int(active_time)} min", border=1, align="C", fill=True)
        pdf.cell(60, 12, f"Intensity: {intensity}", border=1, align="C", fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(10)
        
        pdf.set_font("helvetica", "B", 14)
        pdf.set_text_color(44, 62, 80)
        pdf.cell(0, 10, "AI Clinical Summary", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("helvetica", "", 11)
        pdf.set_text_color(0, 0, 0)
        # Clean text for PDF (fpdf2 core fonts only support latin-1)
        report_text = report.get("report_text", "")
        clean_text = report_text.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 7, clean_text)
        pdf.ln(10)
        
        stats = report.get("stats", {})
        if stats and "totals" in stats:
            pdf.set_font("helvetica", "B", 14)
            pdf.set_text_color(44, 62, 80)
            pdf.cell(0, 10, "Activity Breakdown", new_x="LMARGIN", new_y="NEXT")
            
            totals = stats["totals"]
            labels = list(totals.keys())
            values = list(totals.values())
            
            if sum(values) > 0:
                plt.figure(figsize=(6, 6))
                plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
                plt.title("Daily Activity Distribution")
                
                img_buf = io.BytesIO()
                plt.savefig(img_buf, format='png', bbox_inches='tight')
                img_buf.seek(0)
                plt.close()
                
                pdf.image(img_buf, x=55, w=100)
                pdf.ln(5)
                
                hourly = stats.get("hourly", {})
                if hourly:
                    hours = [str(i) for i in range(24)]
                    plt.figure(figsize=(8, 4))
                    bottoms = [0] * 24
                    
                    colors = ['#5e81ac', '#bf616a', '#d08770', '#a3be8c', '#b48ead', '#ebcb8b', '#88c0d0']
                    color_idx = 0
                    
                    for act in labels:
                        act_mins = [hourly.get(h, {}).get(act, 0) for h in hours]
                        if sum(act_mins) > 0:
                            plt.bar(hours, act_mins, bottom=bottoms, label=act, color=colors[color_idx % len(colors)])
                            bottoms = [bottoms[i] + act_mins[i] for i in range(24)]
                            color_idx += 1
                            
                    plt.title("Hourly Activity Timeline")
                    plt.xlabel("Hour of Day")
                    plt.ylabel("Minutes")
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.tight_layout()
                    
                    img_buf_bar = io.BytesIO()
                    plt.savefig(img_buf_bar, format='png', bbox_inches='tight')
                    img_buf_bar.seek(0)
                    plt.close()
                    
                    pdf.add_page()
                    pdf.set_font("helvetica", "B", 14)
                    pdf.set_text_color(44, 62, 80)
                    pdf.cell(0, 10, "Activity Timeline", new_x="LMARGIN", new_y="NEXT")
                    pdf.image(img_buf_bar, x=10, w=180)
                    pdf.ln(10)
                
            pdf.set_font("helvetica", "B", 12)
            pdf.set_text_color(44, 62, 80)
            pdf.cell(0, 10, "Detailed Minutes:", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("helvetica", "", 11)
            pdf.set_text_color(0, 0, 0)
            for act, mins in totals.items():
                pdf.cell(0, 7, f"- {act}: {mins} mins", new_x="LMARGIN", new_y="NEXT")
                
        pdf_bytes = pdf.output()
        
        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f"Health_Report_{owner_name.replace(' ', '_')}.pdf"
        )
    except Exception as e:
        print(f"PDF Error: {e}")
        return jsonify({"error": str(e)}), 500
