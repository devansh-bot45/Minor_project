# Minor HAR — Human Activity Recognition from Phone Sensors

A web app that recognises human activity from a smartphone's **accelerometer and gyroscope** using a
**Conv1D + Bidirectional LSTM + Temporal Attention** deep-learning model.

- **Live mode** – open the dashboard on a phone, press Start, and see the predicted activity every 1.5 s.
- **Replay mode** – stream one of the project's real recordings through the same pipeline (clearly labelled as *not live*), for demos on a laptop.
- **CSV analysis** – upload a recorded sensor file and get a timeline of predictions.
- **History & reports** – sessions are stored locally (SQLite) with timeline, charts, CSV and PDF export.
- **Model page** – architecture, supported classes and *real* evaluation results.
- *Optional:* the original Google sign-in + MongoDB + Gemini AI daily reports (needs your own credentials).

![Dashboard Preview](docs/images/dashboard_preview.png)

## Supported activities

The backend serves the retrained model **v2** (`backend/model_v2/`) by default. The original model
(`backend/har_model.keras`, unchanged) can be selected with `HAR_MODEL_DIR=.`. Both output **6 classes**:

`Hand Activity`, `Jogging`, `Sports`, `Stairs`, `Still`, `Walking`

At run time the label `Uncertain` is shown when the confidence is below 40 %.
Sleep, upstairs-vs-downstairs and other activities are **not** supported: the model has no such classes.

## How it works

```text
Phone (browser)                                Flask backend                       
───────────────                                ─────────────                       
DeviceMotion API ─► 20 Hz samples              /api/predict                        
[ax ay az] m/s² incl. gravity                  1. validate 60×6 window             
[gx gy gz] deg/s → rad/s                       2. preprocess_window  (same code as training):
every 1.5 s send the last 60 samples ───────►     5 Hz low-pass, remove gravity (0.3 Hz),
(3 s window, 50 % overlap)                        + |accel| + |gyro|  → 60×8           
                                               3. StandardScaler → Conv1D+BiLSTM+Attention → 6 probs
                                               4. rules: near-zero motion → Still,          
                                                  gentle motion can't be Jogging/Stairs/Sports,
                                                  confidence < 40 % → Uncertain              
                                               5. per-session moving average + majority vote
dashboard ◄──────────── activity, confidence, probabilities, which rule (if any) changed the label
                                               6. store prediction in SQLite (backend/data/har.db)
```

The dashboard always shows the model's own top class next to the final label, so it is visible when a
rule (not the network) decided the result.

### Model architecture

```text
Input (60 timesteps × 8 channels)
 ├── Conv1D block 1: 64 filters (k=5, k=3) + BatchNorm + MaxPool + Dropout
 ├── Conv1D block 2: 128 filters (k=3, k=3) + BatchNorm + MaxPool + Dropout
 ├── Conv1D block 3: 256 filters (k=3) + BatchNorm + MaxPool + Dropout
 ├── Bidirectional LSTM (128, return sequences)
 ├── Temporal attention (softmax weights over timesteps)
 ├── Bidirectional LSTM (64)
 ├── Dense(128) + BN + Dropout → Dense(64) + BN + Dropout
 └── Dense(6, softmax)
```
840,647 parameters. Trained with focal loss (γ = 2, α = 0.25) and class weights (`backend/train_model.py`).

## Evaluation (real, reproducible)

### Retrained model v2 vs original model

`backend/compare_models.py` gives both models the same raw windows (full numbers: `backend/model_v2/comparison.md`).
v2 was trained once on data split **by person / recording** before augmentation (see *Retraining*); nothing was tuned on the test data.

| Test data | Windows | Original model | Model v2 |
|---|---|---|---|
| Public-dataset subjects never used for v2 (10 WISDM, 2 Heterogeneity, 9 UCI test subjects) — accuracy / macro F1 | 27,352 | 33.6 % / 0.197 | **77.9 % / 0.796** |
| … Stairs recall on those subjects | 4,571 | 0.000 | **0.928** |
| Phone Stairs session (held out for both), model output | 117 | 0 / 117 | **54 / 117** |
| Phone walking recording `..._174501` (held out for v2, *training data* for the original) | 48 | 100 % | 87.5 % |
| Phone Still / Jogging / Walking recordings used for training by both | 230 | 100 % | 100 % |

The original model saw random windows of all public datasets during training (in other units), so its public-data
numbers are not unseen-person results; they show how it behaves on the correctly formatted input the app sends.
The two phone Stairs files are **one** 95-second session recorded on two devices at the same time; v2 labels
84 % of one file's windows Stairs (dashboard pipeline) but only 17 % of the other's. Phone Stairs recognition is
improved, not solved.

### Original model on the repository recordings

`backend/evaluate.py [model_dir]` runs a model on the labelled phone recordings in this repository using
exactly the dashboard pipeline. Run for the original model (`backend/evaluation_report.md`;
v2: `backend/model_v2/evaluation_report.md`):

| Recording | True label | Used in training? | Windows | Pipeline accuracy |
|---|---|---|---|---|
| still data.csv | Still | yes | 103 | 100 % |
| jogging.csv | Jogging | yes | 64 | 100 % |
| sensor_recording_20260426_162439.csv | Walking | yes | 63 | 100 % |
| sensor_recording_20260427_174501.csv | Walking | yes | 48 | 100 % |
| sensor_recording_20260426_145530.csv | Stairs | **no (held out)** | 63 | **0 %** (predicted mostly Jogging) |
| sensor_recording_20260426_145547.csv | Stairs | **no (held out)** | 54 | **0 %** (predicted mostly Walking) |

**How to read this honestly:** the 100 % rows are recordings the model was trained on, so they only show
that the pipeline is wired correctly. The only held-out data (two files of one Stairs session, 117 windows) is not
recognised at all. The earlier figure of "88.6 % test accuracy" in previous versions of this README could not
be reproduced: `report.txt`, `X_all.npy` and the public datasets are not in the repository.

**Known data issue found during this work:** in `prepare_data.py`, two recordings were added twice with
different labels — `sensor_recording_20260424_131838.csv` is byte-identical to `still data.csv` and
`sensor_recording_20260424_184254.csv` is identical to `jogging.csv`, but both copies were also labelled
**Stairs**. This mislabelled data, mixed units across datasets and a Stairs cap were the causes of the failure
(`MODEL_DIAGNOSIS.md`). Model v2 was trained with the corrected pipeline (see *Retraining*).

## Quick start

Prerequisites: **Python 3.11** (TensorFlow does not support every newer Python yet), **Node.js 18+**.

### 1. Backend

macOS / Linux:
```bash
cd backend
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python app.py
```

Windows (PowerShell):
```powershell
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python app.py
```

The API starts on **http://localhost:5001** (port 5000 is used by AirPlay on macOS). No `.env` file and no
database server are needed — sessions go to `backend/data/har.db`, created automatically.

### 2. Frontend (new terminal)

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. The Vite dev server proxies `/api` to the backend, so only one address is needed.
On a laptop, use **Replay recording** or **Analyse CSV** (laptops have no motion sensor).

### 3. Using a phone's real sensors

Mobile browsers only expose motion sensors on **HTTPS** pages.

```bash
cd frontend
npm run dev:https
```

1. Put the phone and the computer on the same Wi-Fi.
2. Open the `Network: https://<your-ip>:5173` address printed by Vite on the phone.
3. Accept the self-signed certificate warning (development only).
4. Press **Start** (iPhone: allow "Motion & Orientation" when asked).

Alternative: expose `http://localhost:5173` through an HTTPS tunnel such as `cloudflared tunnel --url http://localhost:5173`.

## Configuration

Everything is optional for the core app. Copy the example files if you need to change something:

| File | Variable | Purpose |
|---|---|---|
| `backend/.env` | `PORT` | API port (default 5001) |
| | `HAR_DB_PATH` | SQLite file (default `backend/data/har.db`) |
| | `MONGODB_URI`, `JWT_SECRET`, `GOOGLE_CLIENT_ID` | enable the optional cloud features (all three needed) |
| | `GEMINI_API_KEY` | AI report generation (optional cloud feature) |
| `frontend/.env` | `BACKEND_URL` | where Vite proxies `/api` (default `http://localhost:5001`) |
| | `VITE_API_BASE_URL` | only if the frontend is hosted separately from the backend |
| | `VITE_GOOGLE_CLIENT_ID` | Google sign-in for the optional AI reports |

Optional cloud features: `pip install -r requirements-cloud.txt`, set the variables above, restart. An
**AI Reports** tab then appears. These were part of the original project and were not tested in this
version (no credentials were available).

## CSV formats accepted

1. Android sensor-logger format (the project's recordings): `Timestamp,Sensor Type,Value1,Value2,Value3` — type 1 = accelerometer (m/s²), type 4 = gyroscope (rad/s), timestamps in ms.
2. Wide format: `timestamp_ms,ax,ay,az,gx,gy,gz` (accelerometer m/s², gyroscope rad/s). The **Raw sensor CSV** button after a live session downloads exactly this format.

Data is resampled to 20 Hz. Gaps longer than 1 s are skipped rather than interpolated.

## API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | status, classes, whether cloud features are on |
| GET | `/api/model` | architecture, preprocessing settings, evaluation report |
| POST | `/api/predict` (alias `/predict`) | `{data: 60×6, session_id?, t_sec?}` → prediction |
| POST | `/api/predict/csv` | multipart `file` (+ `save`, `name`) → timeline |
| GET | `/api/samples`, `/api/samples/<id>` | repository recordings for replay |
| POST / GET | `/api/sessions` | create / list sessions |
| GET / DELETE | `/api/sessions/<id>` | session detail / delete |
| POST | `/api/sessions/<id>/end` | finish a session |
| GET | `/api/sessions/<id>/export.csv`, `/report.pdf` | exports |

## Tests

```bash
cd backend && .venv/bin/python -m pytest tests -v     # API, CSV parsing, real-model checks
.venv/bin/python evaluate.py [model_v2]               # regenerate evaluation_report.json/.md for a model
cd ../frontend && npm run lint && npm run build
```

## Project structure

```text
backend/
  app.py              Flask API (core features)
  preprocessing.py    shared signal processing + CSV parsing (used by training AND inference)
  model_service.py    model loading, class mapping checks, rules, per-session smoothing
  storage.py          SQLite sessions & predictions
  reports_pdf.py      session PDF export
  recordings.py       the labelled recordings in the repo root
  evaluate.py         evaluation on those recordings
  cloud_reports.py    OPTIONAL Google sign-in / MongoDB / Gemini reports (original feature)
  train_model.py      training script
  har_model.keras, scaler.pkl, label_encoder.pkl, activity_names.pkl   deployed model artifacts
  backup_model/       an older, unrelated 18-class model (different input shape) — not used
  tests/
frontend/src/
  App.jsx             header, tabs, backend status
  hooks/useSensorStream.js   phone sensors / replay, windows, pause/resume
  views/              Live, Analyse CSV, History, Model
  components/         activity card, timeline, charts, probabilities, (original) Login/Reports
prepare_data.py       builds X_all.npy / y_all.npy from public datasets + recordings
*.csv                 phone recordings (see recordings.py for labels)
```

## Retraining (optional)

Model v2 was built with the corrected pipeline. It needs the three public datasets in one folder
(`<datasets>/uci_har/UCI HAR Dataset/`, `<datasets>/wisdm/wisdm-dataset/raw/phone/`,
`<datasets>/heterogeneity/Phones_accelerometer.csv` + `Phones_gyroscope.csv`; downloads from the UCI repository,
about 1.1 GB for WISDM + Heterogeneity). From the repository root:

```bash
backend/.venv/bin/pip install -r backend/requirements-train.txt
backend/.venv/bin/python prepare_data_v2.py <datasets> <datasets>/windows_v2.npz   # 20 Hz, m/s², split by person
cd backend
MPLBACKEND=Agg .venv/bin/python train_model_v2.py <datasets>/windows_v2.npz 60       # writes model_v2/ (~30 min on CPU)
.venv/bin/python compare_models.py <datasets>/windows_v2.npz                         # v1 vs v2 on the same held-out data
.venv/bin/python evaluate.py model_v2
```

`model_v2/trained_on_recordings.json` lists which phone recordings v2 was trained on (the app shows "held out"
accordingly). `prepare_data.py` / `train_model.py` are kept unchanged as the record of how the original model was built.

## Known limitations

- Phone Stairs recognition is only partly fixed: the only held-out phone Stairs data is one session, and v2's result differs strongly between the two devices that recorded it (84 % vs 17 %).
- v2 regressions: 4 of 48 windows of a held-out phone walking recording are predicted Stairs; on public data some Still windows are predicted Hand Activity (Still recall 0.963 → 0.895).
- `Hand Activity` and `Sports` come only from public datasets; no phone recording in the repo tests them.
- Live capture was tested by replaying real recordings; it has **not** been tested on a physical phone in this version.
- Browser sampling uses a 20 Hz timer; phones may throttle timers when the screen is off (the app requests a screen wake lock).
- iOS and Android can report accelerometer axes with different signs; the training data mostly comes from Android.
- Time per activity is estimated as 1.5 s per prediction window.
- Not a medical device; predictions can be wrong.

## Contributors

- Vansh Tambi
- Vivek Pasi
