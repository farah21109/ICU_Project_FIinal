# MedWatch ICU

ICU early-warning dashboard that ranks patients by LSTM deterioration risk and shows live vital signs.

> **Student project. Not for clinical use.** It runs on a research time-series dataset and is not connected to any hospital system.

## Run

```powershell
node server.js          # website  -> http://127.0.0.1:8000
python model_api.py     # LSTM API -> http://127.0.0.1:5000  (needs: pip install -r requirements-model.txt)
```

No internet needed: fonts and Chart.js are bundled in `fonts/` and `vendor/`.
`single code.html` is a standalone copy that opens by double-click (rebuild it with `python build_single.py`).
The trained model is included in `model/` (`icu_lstm_model.h5`, `scaler.pkl`, `feature_cols.pkl`): input is 10 time steps x 17 features, output is a risk probability. See `MODEL_INTEGRATION.md`. The top bar shows whether it is live, using fallback scoring, or offline.

## If the model is not working

The top bar shows the model state, and Settings > System shows the reason.

| Top bar says | Meaning | Fix |
| --- | --- | --- |
| Model API offline | `model_api.py` is not running | Run `python model_api.py` and wait for "LSTM model loaded OK" (15-30 s) |
| Fallback scoring | API is up but the LSTM did not load | Read the reason in the terminal or Settings. Usually a missing package: `pip install -r requirements-model.txt` |
| LSTM model live | Working | |

Other causes: Python 3.13 or newer (TensorFlow may not support it yet, use Python 3.10 to 3.12), and the three files must be in `model/` with these exact names: `icu_lstm_model.h5`, `scaler.pkl`, `feature_cols.pkl`.

## Files

| File | Purpose |
| --- | --- |
| `index.html` | Page structure (login, dashboard, patients, alerts, reports, settings) |
| `styles.css` | Design system: thermal palette, radar stage, dossier, responsive layout |
| `prepare_data.py`, `data/` | Builds `data/icu-data.js` (real patients, model scores, metrics) from the dataset and trained model |
| `app.js` | Data, model calls, radar, replay, body figure, charts, navigation |
| `server.js` | Static file server |
| `model_api.py`, `model/` | LSTM prediction API and model files |
| `ICU_Patient_Dataset.xlsx` | Earlier 200-patient registry (not used by the model) |

## Design notes

- Concept: "Thermal ward". The ward is a radar where distance from the centre is LSTM risk, so the sickest patients sit closest to the middle. Heat is the only colour: cold blue is stable, violet to amber is watch, red is critical.
- Replay: the scrubber under the radar replays the last 24 readings. The whole ward drifts, and the counts and priority list update as risk changes.
- Patient dossier: opening a patient zooms in from the dot you clicked. It shows a body figure whose organ systems light up from the vitals and labs (the heart beats at the patient's heart rate, the lungs breathe at their respiratory rate), a live ECG and pleth trace, a risk gauge, trends, labs and notes.
- Login: a boot sequence (flatline, heartbeat, ripple, wordmark) that opens into the ward.
- Type: Anybody (variable width, used for numerals and headings), Instrument Sans (interface), JetBrains Mono (IDs and data).
- Calm mode (Settings, Display, or the system "reduce motion" setting) switches off the sweep, pulses and transitions.

## Data and results

- **Patients are real rows** from `data/icu_timeseries_dataset.xlsx` (218 patients, about 42 hourly readings each). The site shows each patient's last 24 readings.
- **Risk history is the model's own output.** `prepare_data.py` runs the trained LSTM over every 10-reading window, so the replay scrubber and risk charts show real scores. The first readings of a short stay have no score yet (the model needs 10 hours), and they are drawn dashed.
- **Live scoring** sends each patient's exact last 10 readings (17 features) to `model_api.py`. It matches the precomputed scores to within 5e-8.
- **Model performance panel** (Insights) uses the same patient-level split as the training notebook: 44 held-out test patients, 1,468 sequences. ROC-AUC 0.949, accuracy 96%, high-risk recall 0.89, precision 0.95. A logistic regression on the latest reading scores 0.933 on the same patients.
- Each patient page says whether the patient was in the training set or held out. Scores for training patients are not an independent test.
- **Not in the dataset, so simulated for display:** wards and beds. The ECG and pleth traces are drawn from the heart rate and are not recorded waveforms.
- The model is very confident (96 patients score about 0.9999), so inside the red core of the radar, distance from the centre shows rank, not exact score. The 0.35 and 0.65 rings are exact. Scores of 0.995 and above are shown as "0.99+".
- The site's "high risk" band is above 0.65. The metrics above use the notebook's 0.5 threshold.
- To rebuild the data after retraining: `python prepare_data.py`.
# ICU_Project_FIinal
