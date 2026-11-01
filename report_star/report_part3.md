## CHAPTER 7: IMPLEMENTATION (TOOLS AND TECHNOLOGY USED)

### 7.1 Overview of Technology Stack

The system consists of a React frontend, a Flask backend, a TensorFlow/Keras model and a local SQLite database. Table 5 lists the main tools; the versions are those installed on the development laptop (Apple M3, 16 GB RAM, macOS).

**Table 5: Tools and Technologies Used**

| Layer | Technology | Version | Purpose in this project |
|-------|-----------|---------|-------------------------|
| Frontend | React, Vite, Chart.js | 19.3.1, 8.3.5, 4.6.1 | Browser client, development server with HTTPS, live charts |
| Backend framework | Flask, Flask-CORS | 3.1.3, 6.0.5 | REST API |
| Deep learning | TensorFlow, Keras | 2.21.0, 3.15.1 | Training and running the model |
| Signal processing | SciPy | 1.17.1 | Butterworth filters, resampling |
| Data processing | NumPy, pandas | 2.4.6, 3.1.0 | Arrays and CSV parsing |
| Machine learning utilities | scikit-learn | 1.8.0 | StandardScaler, LabelEncoder, metrics (same version that saved the scaler and encoder) |
| Storage and export | SQLite (Python standard library), fpdf2 | built in, 2.8.9 | Sessions, predictions and PDF reports |
| Testing | pytest | 9.1.1 | Automated backend tests |
| Languages | Python 3.11, JavaScript (ES2022) | — | Backend and machine learning; frontend |

The Python packages are listed in backend/requirements.txt, requirements-train.txt (training scripts) and requirements-cloud.txt (optional module), and the frontend packages in frontend/package.json.

### 7.2 Frontend–Backend Communication

The browser client exists to stream sensor data to the model and display its output. The hook useSensorStream.js reads accelerationIncludingGravity (m/s²) and rotationRate (converted from degrees per second to rad/s) from the DeviceMotion interface, which mobile browsers expose only to HTTPS pages and, on iOS, only after the user grants permission. A 50 ms timer appends the latest reading as one sample, so the live path samples at a fixed 20 Hz, whereas uploaded CSV files are resampled by timestamp on the server. The client keeps the last 60 samples and posts a window to /api/predict after the first 60 samples and then every 30 samples, with at most three requests in flight. The returned label, confidence, network class and class probabilities are shown on a live view with an activity timeline; other views analyse uploaded CSV files, list stored sessions with CSV and PDF export, and show the served model's settings and evaluation report. Stored recordings can be replayed through the same loop for demonstrations, clearly marked as not live. The Vite development server forwards /api requests to Flask (port 5001), so a phone on the same network needs only one address.

### 7.3 Backend Implementation

At start-up (create_app in backend/app.py) the model service loads the Keras network, label encoder, activity mapping and fitted scaler from the model folder and refuses to start unless they are consistent: the encoder classes must equal the activity names, the network must accept 60 × 8 input and output six classes, and the scaler must have eight features. backend/preprocessing.py provides the shared preprocess_window function (Algorithm 2), window validation and a CSV parser for the Android sensor-logger and dashboard formats. /api/predict implements Algorithm 3 with smoothing state kept per session in memory and protected by a lock, and /api/predict/csv applies the same pipeline to every window of an uploaded file. Further endpoints provide health and model information, replay recordings, session management and CSV and PDF exports. Sessions and predictions are stored in two SQLite tables, sessions and predictions; each prediction row holds the window time, final activity, network class, confidence, applied rule and all class probabilities.

The evaluation scripts backend/evaluate.py and backend/compare_models.py run the model through the application's own code on the phone recordings and the held-out windows and produce all results of Chapter 8 (evaluation_report.json, comparison.json). The 25 pytest tests in backend/tests/ cover the API with a stub model (valid and invalid input, the stationary rule, sessions, CSV formats, gaps and exports) and check the preprocessing and the real model files (consistency, probabilities summing to one, correct classification of the Still and Jogging recordings); all 25 passed on 29 October 2026.
