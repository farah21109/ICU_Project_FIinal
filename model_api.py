from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path

import numpy as np

APP_DIR = Path(__file__).parent
MODEL_DIR = APP_DIR / "model"
SEQ_LEN = 10

DEFAULT_FEATURE_COLS = [
    "Age",
    "Gender",
    "Heart_Rate",
    "Systolic_BP",
    "Diastolic_BP",
    "SpO2",
    "Respiratory_Rate",
    "Temperature",
    "WBC",
    "Lactate",
    "Creatinine",
    "Platelets",
    "Hemoglobin",
    "Glucose",
    "Sodium",
    "Potassium",
    "BUN",
]

model = None
scaler = None
feature_cols = DEFAULT_FEATURE_COLS
load_error = None


def load_artifacts():
    global model, scaler, feature_cols, load_error
    if (model is not None and scaler is not None) or load_error is not None:
        return

    try:
        model_path = MODEL_DIR / "icu_lstm_model.h5"
        if not model_path.exists():
            model_path = MODEL_DIR / "icu_lstm_model.keras"

        scaler_path = MODEL_DIR / "scaler.pkl"
        features_path = MODEL_DIR / "feature_cols.pkl"

        if not model_path.exists() or not scaler_path.exists():
            load_error = "Model files not found in the model/ folder; using clinical fallback score."
            return

        import joblib
        import tensorflow as tf

        loaded_model = tf.keras.models.load_model(model_path)
        loaded_scaler = joblib.load(scaler_path)  # needs scikit-learn installed
        loaded_cols = joblib.load(features_path) if features_path.exists() else feature_cols

        # Only publish once everything loaded, so "model_loaded" is never half true.
        model, scaler, feature_cols = loaded_model, loaded_scaler, loaded_cols
    except ModuleNotFoundError as exc:
        load_error = f"Missing Python package '{exc.name}'. Run: pip install -r requirements-model.txt"
    except Exception as exc:
        load_error = f"Could not load the model: {exc}"


def read_feature(record, column):
    aliases = {
        "Gender": "Gender",
        "Heart_Rate": "Heart_Rate",
        "Systolic_BP": "Systolic_BP",
        "Diastolic_BP": "Diastolic_BP",
        "SpO2": "SpO2",
        "Respiratory_Rate": "Respiratory_Rate",
        "Hemoglobin": "Hemoglobin",
    }
    value = record.get(column, record.get(aliases.get(column, column), 0))
    if column == "Gender" and isinstance(value, str):
        return 1 if value.upper().startswith("M") else 0
    return float(value)


def fallback_score(records):
    latest = records[-1]
    score = 0.08
    spo2 = float(latest.get("SpO2", 98))
    rr = float(latest.get("Respiratory_Rate", 16))
    hr = float(latest.get("Heart_Rate", 80))
    sbp = float(latest.get("Systolic_BP", 120))
    lactate = float(latest.get("Lactate", 1.2))
    temp = float(latest.get("Temperature", 37))

    if spo2 < 92:
        score += (92 - spo2) * 0.045
    if rr > 22:
        score += (rr - 22) * 0.035
    if hr > 100:
        score += (hr - 100) * 0.012
    if sbp < 100:
        score += (100 - sbp) * 0.014
    if lactate > 2:
        score += (lactate - 2) * 0.18
    if temp > 38:
        score += (temp - 38) * 0.12

    return max(0.0, min(1.0, score))


def predict(records):
    load_artifacts()
    latest_records = records[-SEQ_LEN:]

    if model is None or scaler is None:
        probability = fallback_score(latest_records)
    else:
        features = np.array(
            [[read_feature(record, col) for col in feature_cols] for record in latest_records],
            dtype=float,
        )
        scaled = scaler.transform(features)
        model_input = scaled.reshape(1, SEQ_LEN, len(feature_cols))
        probability = float(model.predict(model_input, verbose=0)[0][0])

    if probability >= 0.65:
        status = "High Risk"
    elif probability >= 0.35:
        status = "Medium Risk"
    else:
        status = "Low Risk"

    return {
        "risk_probability": probability,
        "predicted_risk": 1 if probability >= 0.5 else 0,
        "risk_status": status,
        "model_loaded": model is not None and scaler is not None,
        "message": load_error,
    }


class Handler(BaseHTTPRequestHandler):
    def send_json(self, status, body):
        data = json.dumps(body).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            pass  # the browser stopped waiting; nothing to do

    def do_OPTIONS(self):
        self.send_json(200, {"ok": True})

    def do_GET(self):
        if self.path == "/health":
            load_artifacts()
            self.send_json(200, {"status": "ok", "model_loaded": model is not None and scaler is not None, "message": load_error})
            return
        if self.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        self.send_json(404, {"error": "Not found. Try /health"})

    def do_POST(self):
        if self.path != "/predict":
            self.send_json(404, {"error": "Not found"})
            return

        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        records = body.get("records", [])

        if len(records) < SEQ_LEN:
            self.send_json(400, {"error": f"At least {SEQ_LEN} records are required."})
            return

        self.send_json(200, predict(records))


if __name__ == "__main__":
    print("Loading model (this can take 15-30 seconds the first time)...")
    load_artifacts()
    if model is not None and scaler is not None:
        print("LSTM model loaded OK.")
    else:
        print("WARNING: LSTM model NOT loaded -> using fallback score.")
        print("Reason:", load_error)
    server = HTTPServer(("127.0.0.1", 5000), Handler)
    print("Model API running at http://127.0.0.1:5000")
    server.serve_forever()
