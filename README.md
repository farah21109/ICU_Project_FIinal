# MedWatch ICU

An ICU early-warning dashboard. An LSTM reads the last 10 hours of a patient's vital signs and lab values and predicts whether they are likely to deteriorate in the next hour. The ward is shown as a radar where **the closer a patient is to the centre, the higher their predicted risk**.

**Live demo: [icu-website.onrender.com](https://icu-website.onrender.com/)**

> **Student project. Not for clinical use.** It runs on a research time-series dataset and is not connected to any hospital system.

<!--
Add screenshots here once they are in the repo, for example:
![Ward radar](docs/ward-radar.png)
![Patient dossier](docs/patient-dossier.png)
![Model performance](docs/model-performance.png)
-->

## Try the live demo

1. Open the link above and click **Enter the ward**. The sign-in is a demo, so the pre-filled details work.
2. On the radar, drag the scrubber at the bottom (or press play) to replay the last 24 readings, then click any dot to open that patient.
3. Open **Insights** to see the model's measured performance.

The demo is hosted on a free plan, which has two effects:
- **The first load can be slow.** The model API sleeps when idle and takes up to a minute or two to wake. While it wakes, the site shows saved scores from the same trained LSTM, so nothing is missing. When the API is awake, the top-bar indicator reads **LSTM model live** and patients are scored live.
- If the indicator never turns live, wait a minute and press the refresh button in the top bar.

## Results

Evaluated on **44 held-out patients (1,468 sequences)** that the model never saw in training. The split is by patient, so no patient appears in both train and test.

| Metric | Result |
| --- | --- |
| ROC-AUC | **0.949** |
| Accuracy | 96% |
| High-risk precision | 0.95 |
| High-risk recall | 0.89 |
| High-risk F1 | 0.92 |

Confusion matrix at threshold 0.5: 1,020 true negatives, 19 false alarms, 47 missed, 382 caught.

A logistic regression on the latest reading alone scores 0.933 ROC-AUC on the same patients, so the LSTM's use of the last 10 hours adds about 0.016.

## Features

**Ward radar**
- Every patient is a node, placed by ward and bed. Distance from the centre shows LSTM risk, and nodes pulse at the patient's heart rate.
- A replay scrubber steps through the last 24 readings, and the whole ward drifts as risk changes. Counts and the priority list update with it.
- A priority queue ranks the patients who need attention first. Hovering a row highlights the patient on the radar.

**Patient dossier**
- A body figure whose organ systems (respiratory, cardiovascular, renal, metabolic, blood and circulation, temperature) light up according to vitals and labs.
- Live-style ECG and pleth traces, drawn from the patient's heart rate.
- Risk gauge, five vital signs with Normal, Watch or Abnormal flags, risk-horizon chart, trend charts, lab values and a "What changed" summary computed from the last 10 readings.
- States whether the patient was in the training set or held out.

**Other pages**
- **Patients:** searchable and sortable list, plus a bed map.
- **Alerts:** a timeline of patients above the warning threshold, grouped by how long they have been above it.
- **Insights:** population statistics, risk distribution, risk by ward and age group, and a **Model performance** panel with the ROC curve and confusion matrix.
- **Settings:** profile, "edited by" staff picker with history, display options (calm mode, compact lists) and system status.

**Also:** a model-status indicator (live, fallback or offline), responsive layout down to phones, keyboard-accessible lists and scrubber, a calm mode that switches off motion, and fully offline assets (fonts and Chart.js are bundled).

## How it works

```
data/icu_timeseries_dataset.xlsx --prepare_data.py--> data/icu-data.js ----+
                                                                           v
model/icu_lstm_model.h5, scaler.pkl, feature_cols.pkl            Browser (index.html + app.js)
                          |                                                ^
                          +--------> model_api.py  <---- POST /predict ----+
```

- `prepare_data.py` runs the trained model over every 10-reading window of every patient, so the replay and risk charts show the model's own scores. It also reproduces the training notebook's test split to compute the metrics.
- The website works **without** the API, using those precomputed scores. When the API is reachable, the refresh button and patient pages score patients live, sending their exact last 10 readings.
- Live scores match the precomputed ones to within 5e-8.

### The model

| | |
| --- | --- |
| Input | 10 hourly readings x 17 features |
| Features | Age, Gender, Heart_Rate, Systolic_BP, Diastolic_BP, SpO2, Respiratory_Rate, Temperature, WBC, Lactate, Creatinine, Platelets, Hemoglobin, Glucose, Sodium, Potassium, BUN |
| Architecture | LSTM(64) -> Dropout -> BatchNorm -> LSTM(32) -> Dropout -> Dense(16) -> Dense(1, sigmoid), 34,209 parameters |
| Scaling | MinMaxScaler, fitted on training patients only |
| Target | The risk label for the hour **after** the 10-hour window (next-hour prediction) |
| Split | Patient-level, 80/20 and stratified: 174 train and 44 test patients |
| Training | Class weights, with the best epoch chosen by validation AUC |

### The data

`data/icu_timeseries_dataset.xlsx`: 9,178 hourly rows, 218 patients (about 42 readings each), 22 columns. About 23.5% of rows are labelled high risk.

**Source:** [add where the dataset comes from, and say if it is synthetic]

## Deployment

The project runs as two separate services on Render's free plan:

| Service | What it is | Address |
| --- | --- | --- |
| Website | The static front end (`index.html`, `styles.css`, `app.js`, `data/`, `fonts/`, `vendor/`) | https://icu-website.onrender.com/ |
| Model API | `model_api.py`, which loads the LSTM and scores patients | https://icu-model-api.onrender.com |

- **Why two services:** the model needs TensorFlow, which is too large for serverless hosts such as Vercel. The website itself is static and can be hosted anywhere.
- **Connecting them:** `MODEL_API_URL` and `MODEL_HEALTH_URL` at the top of `app.js` hold the API's `https` address. The API allows cross-origin requests, so the browser can call it from the website's address.
- **Check the API:** open `https://icu-model-api.onrender.com/health`. It should return `"model_loaded": true`. If it returns `false`, the `message` field says why.
- **Redeploying:** push to the repository, and each service redeploys. After changing `app.js` or `styles.css`, hard refresh the browser with `Ctrl + Shift + R`.
- **Free-plan behaviour:** services sleep when idle, and a cold start can take a minute or more because TensorFlow has to load. The free plan has limited memory, so if the API crashes on startup, check its logs and use `tensorflow-cpu` in `requirements-model.txt`.

## Run it locally

You need **Node.js** and **Python 3.10 to 3.12** (TensorFlow may not support newer versions yet).

```bash
# one-time setup
pip install -r requirements-model.txt

# terminal 1: model API  (wait for "LSTM model loaded OK", it can take up to a minute)
python model_api.py

# terminal 2: website
node server.js
```

Open **http://127.0.0.1:8000**. Locally, the API listens on `http://127.0.0.1:5000`. If `app.js` currently points at the hosted API, set `MODEL_API_URL` and `MODEL_HEALTH_URL` at the top of the file back to `http://127.0.0.1:5000/predict` and `http://127.0.0.1:5000/health` while testing locally, and change them back before deploying.

You can also open `single code.html` by double-clicking it. It is a standalone copy of the site, and it uses the precomputed scores.

### Troubleshooting

| Top bar says | Meaning | Fix |
| --- | --- | --- |
| Model API offline | The API is not running or not reachable | Locally, start `model_api.py` and wait for "LSTM model loaded OK". When hosted, wait for it to wake, then refresh. |
| Fallback scoring | The API is up but the LSTM did not load | Read the reason in the terminal, the host's logs, or Settings, System. Usually a missing package: `pip install -r requirements-model.txt` |
| LSTM model live | Working | |

The three files in `model/` must keep these exact names: `icu_lstm_model.h5`, `scaler.pkl`, `feature_cols.pkl`.

### API

| Endpoint | Description |
| --- | --- |
| `GET /health` | `{"status": "ok", "model_loaded": true, "message": null}`. `message` explains why if the model did not load. |
| `POST /predict` | Body: `{"patient_id": "P-001", "records": [10 objects, each with the 17 features]}`. Returns `risk_probability`, `risk_status` and `model_loaded`. |

Details are in `MODEL_INTEGRATION.md`.

## Project structure

```
index.html, styles.css, app.js    The website
server.js                         Static file server for local use (port 8000)
model_api.py                      LSTM prediction API (port 5000 locally)
prepare_data.py                   Builds data/icu-data.js from the dataset and model
build_single.py                   Builds the standalone "single code.html"
requirements-model.txt            Python packages for the API
data/                             Time-series dataset and the generated icu-data.js
model/                            Trained model, scaler and feature list
fonts/, vendor/                   Bundled fonts and Chart.js, so it works offline
ICU_Patient_Dataset.xlsx          Earlier 200-patient registry (not used by the model)
```

After retraining the model, run `python prepare_data.py` to refresh the site's data and metrics, then `python build_single.py` to refresh the standalone copy.

## Limitations

- **Not clinically validated.** It was tested on one dataset and one patient-level split of 44 test patients, so the metrics could shift noticeably with a different split.
- **Wards and beds are simulated for display**, since the dataset has none. The ECG and pleth traces are drawn from the heart rate and are not recorded waveforms.
- **The model is very confident.** About 96 patients score near 0.9999, so inside the red core of the radar, distance from the centre shows rank rather than exact score. The 0.35 and 0.65 rings are exact, and scores of 0.995 and above are shown as "0.99+".
- **Thresholds differ between the site and the metrics.** The site calls a patient high risk above 0.65 and medium from 0.35. The metrics above use the notebook's 0.5 threshold.
- **Prediction horizon is one hour**, not several.
- Sign-in is a demo and accepts any input. There is no database.

## Built with

TensorFlow/Keras, scikit-learn, pandas, Python, Node.js, vanilla JavaScript, Chart.js, Render, and the Anybody, Instrument Sans and JetBrains Mono fonts.


