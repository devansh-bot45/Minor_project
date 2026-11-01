# Minor HAR - System Architecture

> Visual overview of system components, data flow, and training pipelines.

## System Architecture

The following diagram illustrates the real-time operational flow of the system. It shows how sensor data is captured on the frontend, processed by the Flask backend, classified by the hybrid deep learning model, and stored in a local SQLite database. Google sign-in, MongoDB and Gemini reports are an optional module (dashed boxes) that is enabled only when credentials are configured.

```mermaid
graph TB
    classDef frontend fill:#E3F2FD,stroke:#1565C0,stroke-width:2px,color:#0D47A1;
    classDef backend fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px,color:#1B5E20;
    classDef ml fill:#FFF3E0,stroke:#EF6C00,stroke-width:2px,color:#E65100;
    classDef db fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px,color:#4A148C;
    classDef api fill:#FFEBEE,stroke:#C62828,stroke-width:2px,color:#B71C1C;

    subgraph "Frontend Architecture (React 19 & Vite)"
        UI["📱 User Interface<br/>(Live, Analyse CSV, History, Model)"]:::frontend
        SensorCapture["⏱️ Sensor Capture Engine<br/>(Acc & Gyro @ 20Hz)"]:::frontend
        Buffer["📦 Sliding Window<br/>(60 samples, sent every 1.5 s)"]:::frontend
        AuthFront["🔐 Optional Google OAuth Login"]:::frontend

        SensorCapture --> Buffer
        AuthFront -.-> UI
    end

    subgraph "Backend API (Flask & PyJWT)"
        Gateway["🌐 API Gateway<br/>(REST Endpoints)"]:::backend
        Preprocess["⚙️ Feature Engineering<br/>(Scale, Filter, Extract)"]:::backend
        Smoothing["📊 Rules + Per-Session Smoothing<br/>(threshold, EMA, majority vote)"]:::backend
        GeminiService["🤖 Optional AI Reporting Service"]:::backend
        
        Gateway --> Preprocess
    end

    subgraph "Deep Learning Engine"
        ModelInference["🧠 Hybrid Model Inference<br/>(Conv1D + BiLSTM + Attention)"]:::ml
        ModelArtifacts["💾 Trained Artifacts<br/>(har_model.keras, scaler.pkl)"]:::ml
        
        ModelArtifacts -.-> ModelInference
        Preprocess --> ModelInference
        ModelInference --> Smoothing
    end

    subgraph "Storage & External Services"
        SQLite[("🗄️ SQLite<br/>(Sessions & Predictions)")]:::db
        MongoDB[("🗄️ Optional MongoDB Atlas<br/>(Activity Logs & Reports)")]:::db
        GeminiAPI{{"☁️ Google Gemini API<br/>(LLM Integration)"}}:::api
        OAuthAPI{{"☁️ Google Auth API"}}:::api
    end

    %% Connections
    Buffer -- "POST /api/predict" --> Gateway
    UI -- "JWT Auth Calls" --> Gateway
    AuthFront <--> OAuthAPI

    Smoothing --> SQLite
    Smoothing -.-> MongoDB
    GeminiService <--> GeminiAPI
    GeminiService --> MongoDB
    
    Gateway --> GeminiService
```

## Data & Training Pipeline

The offline training pipeline is responsible for merging multiple academic datasets with custom mobile recordings, engineering magnitude features, applying data augmentation, and training the `Conv1D + BiLSTM + Attention` model.

```mermaid
graph TD
    classDef data fill:#E1F5FE,stroke:#0277BD,stroke-width:2px,color:#01579B;
    classDef process fill:#FBE9E7,stroke:#D84315,stroke-width:2px,color:#BF360C;
    classDef output fill:#F1F8E9,stroke:#33691E,stroke-width:2px,color:#1B5E20;

    subgraph "Data Sources"
        Wisdm["WISDM Dataset"]:::data
        Uci["UCI HAR Dataset"]:::data
        Hetero["Heterogeneity Dataset"]:::data
        Custom["Custom Mobile CSVs"]:::data
    end

    subgraph "Data Preparation (prepare_data.py)"
        Filter["Butterworth Low-pass Filter"]:::process
        Sep["Gravity/Body Accel Separation"]:::process
        Mag["Magnitude Feature Engineering"]:::process
        Window["Sliding Window (Size 60, Step 30)"]:::process
        Augment["Data Augmentation (Jitter, Scale, Shift, Warp, Permute, Invert)"]:::process
        
        Wisdm & Uci & Hetero --> Filter
        Custom --> Filter
        Filter --> Sep --> Mag --> Window
        Custom -.-> Augment -.-> Window
    end

    subgraph "Model Training (train_model.py)"
        Scale["Global Feature Scaling"]:::process
        Train["Train Conv1D + BiLSTM + Attention Model<br/>(focal loss)"]:::process
        ClassWeights["Apply Class Weights"]:::process
        
        Window -->|X_all.npy, y_all.npy| Scale
        Scale --> Train
        ClassWeights -.-> Train
    end

    subgraph "Artifacts Generation"
        Keras["har_model.keras"]:::output
        Scaler["scaler.pkl"]:::output
        Encoder["label_encoder.pkl"]:::output
    end

    Train --> Keras & Scaler & Encoder
```

---

## Implementation

> Tools, technologies, and development details

### Tech Stack

| Category | Technologies Used | Details |
| :--- | :--- | :--- |
| **Frontend Language** | JavaScript, JSX, HTML5, CSS3 | Used for the web application UI and mobile browser sensor capture via Web APIs. |
| **Frontend Framework** | React 19, Vite | Fast, modern client-side framework with component-driven architecture. |
| **Backend Language** | Python 3.11 | Primary language for API routing, data processing, and ML training pipelines. |
| **Backend Framework** | Flask, Flask-CORS | Lightweight Python microframework serving the REST API and prediction endpoints. |
| **Machine Learning** | TensorFlow / Keras, Scikit-learn | Used for defining, training, and running inference on the Conv1D + BiLSTM + Attention model. |
| **Data Processing** | NumPy, SciPy, Pandas | Used heavily in the data preparation and runtime feature engineering. |
| **Database** | SQLite (built into Python) | Local storage of recording sessions and predictions; no setup required. |
| **Optional Database** | MongoDB Atlas (PyMongo) | Activity logs and health reports for the optional cloud module. |
| **Authentication (optional)** | Google OAuth, PyJWT | User login exchanging Google tokens for signed JWTs. |
| **External APIs (optional)** | Google Gemini API | AI-generated daily health summaries (not tested in the final version). |
| **Visualizations** | Chart.js, Framer Motion | Interactive timeline graphs and smooth UI animations on the React client. |

### Core Project Flow

1. **Capture**: The React frontend (`useSensorStream.js`) reads the phone's accelerometer and gyroscope at **20 Hz** (HTTPS page required on phones). A recorded file can be replayed instead; the dashboard labels this clearly as not live.
2. **Window**: The last **60 samples (3 seconds)** are sent every **30 new samples (1.5 seconds)**.
3. **Transmit**: Each window is sent over HTTP to the Flask `POST /api/predict` endpoint.
4. **Engineer**: Flask applies the same zero-phase Butterworth filtering and magnitude features as the training data, giving a `(1, 60, 8)` tensor.
5. **Infer**: The pre-trained Keras model outputs probabilities for 6 classes (Hand Activity, Jogging, Sports, Stairs, Still, Walking).
6. **Post-process**: Rules (near-zero motion → Still, confidence < 40% → Uncertain, intensity gating) and per-session smoothing (moving average + majority vote over 5 windows) stabilise the label.
7. **Store**: Predictions are saved per session in SQLite and can be viewed in History and exported as CSV or PDF. If the optional cloud module is configured, signed-in users' predictions are also logged to MongoDB for Gemini daily reports.
