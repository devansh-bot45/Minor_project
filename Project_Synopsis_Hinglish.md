# Project Synopsis: Human Activity Recognition (HAR) System using Hybrid Deep Learning

**Project Title:** Minor_HAR - Real-time Activity Tracking & AI Health Reporting
**Team Members:** Vansh Tambi, Vivek Pasi

---

### 1. Introduction (Project ka Basic Idea)
Yeh project ek real-time **Human Activity Recognition (HAR)** system hai jo mobile sensors (accelerometer aur gyroscope) ka use karke user ki activities ko detect karta hai. Iska main goal user ki activities ko record karna, unki history save karna, aur timeline / CSV / PDF report ke through dikhana hai. Gemini AI se daily health report banana ek **optional** feature hai (credentials ke bina band rehta hai, aur abhi tak test nahi hua).

### 2. Problem Statement (Humein iski zaroorat kyun hai?)
Aaj kal ke lifestyle mein log apni physical activity track nahi kar paate. Existing apps ya toh expensive wearables mangte hain ya phir utne accurate nahi hote. Humein ek aisa solution chahiye tha jo normal mobile browser se hi data capture kare aur activities classify kar sake.

### 3. Objectives (Main Goal kya hai?)
- Mobile sensor data ko real-time mein capture karna.
- **Hybrid Deep Learning Model** (Conv1D + BiLSTM + Attention) ka use karke 6 activities (Walking, Jogging, Stairs, Still, Hand Activity, Sports) classify karna, aur accuracy ko reproducible tarike se measure karna. (Measured: repository ki 395 labelled phone windows par 70.4% accuracy, lekin inme se 278 windows training data ka hissa thi; held-out Stairs recordings par recall 0% hai. Pehle likha gaya 88.6% reproduce nahi ho saka.)
- Har recording session aur uski predictions ko local **SQLite** database mein save karna, aur history, CSV aur PDF export dena.
- **Optional:** Google sign-in, MongoDB activity logs aur Gemini AI daily summaries — yeh code mein hain lekin credentials ke bina band rehte hain aur abhi tak verify nahi hue.

### 4. Methodology (Kaise Kaam Karta Hai?)
Is project ko humne 3 parts mein divide kiya hai:

1.  **Frontend (React 19 + Vite):**
    - Mobile browser se accelerometer aur gyroscope ka data capture karta hai (20 Hz frequency par). Phone par sensors ke liye page HTTPS par khulna zaroori hai (`npm run dev:https`).
    - Har 1.5 second mein last 60 samples (3 seconds ki window) backend ko bhejta hai.
    - Laptop par demo ke liye repository ki real recordings ko "replay" kar sakte hain (screen par saaf likha aata hai ki yeh live data nahi hai), ya CSV file upload kar sakte hain.
2.  **Backend (Flask):**
    - Received data ko wahi preprocessing deta hai jo training mein use hui thi (Butterworth filter, gravity alag karna, magnitude features → 60 x 8 window).
    - ML model se activity predict karwata hai, phir rules aur per-session smoothing lagata hai.
    - Predictions ko SQLite (`backend/data/har.db`) mein save karta hai.
3.  **Deep Learning Model:**
    - **Conv1D (3 blocks):** Local motion patterns extract karne ke liye.
    - **BiLSTM:** Time-series patterns ko dono directions mein samajhne ke liye.
    - **Temporal Attention:** Important time steps par focus karne ke liye.

### 5. Technology Stack (Software aur Tools)
- **Frontend:** React.js, Vite, Chart.js, Framer Motion.
- **Backend:** Flask (Python 3.11).
- **Machine Learning:** TensorFlow/Keras, NumPy, SciPy, Scikit-learn.
- **Database (primary):** SQLite — sessions aur predictions store karne ke liye; koi setup ya credentials nahi chahiye.
- **Optional (verify nahi hua):** Google OAuth + PyJWT (login), MongoDB Atlas (activity logs), Google Gemini API (AI reports).

### 6. Key Features (Main Highlights)
- **Real-time Detection:** Har 1.5 second mein activity predict karta hai. Still, Walking aur Jogging ki training recordings par predictions sahi hain; Stairs abhi held-out recordings par detect nahi hota.
- **Session History:** Har session ki timeline, activity-wise time, aur confidence chart; CSV aur PDF export.
- **CSV Analysis aur Replay:** Recorded sensor file upload karke ya repository recording replay karke bina phone ke bhi demo ho sakta hai.
- **Cross-Platform:** Mobile browser par directly chalne ke liye banaya gaya hai, kisi app installation ki zaroorat nahi. (Real phone par live sensors abhi test nahi hue.)
- **Optional AI Health Reports:** Gemini AI se din ki summary aur report sharing — sirf tab jab Google, MongoDB aur Gemini credentials configure kiye jaayein; abhi tak test nahi hua.

### 7. Future Scope (Aage kya ho sakta hai?)
- Naye labelled phone recordings (khaas kar Stairs) record karke model ko sahi labels aur leakage-free evaluation ke saath retrain karna.
- Isme heart rate aur sleep patterns ka data bhi add kiya ja sakta hai.
- More specific activities jaise Cycling ya Gym exercises ko recognize karna.
- Diet recommendation system integrate karna based on activity intensity.

---
