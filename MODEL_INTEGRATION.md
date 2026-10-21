# Model Integration

The website now calls a local model API at:

```text
http://127.0.0.1:5000/predict
```

Authentication and the existing dashboard pages were left unchanged. The only frontend behavior added is:

- after login, the dashboard tries to refresh top-risk LSTM scores from the API
- the existing Refresh button retries the model API
- opening a patient detail page refreshes that patient's LSTM score from the API

## Save These Files From The Notebook

After training the LSTM model, run:

```python
model.save("icu_lstm_model.h5")

import joblib
joblib.dump(scaler, "scaler.pkl")
joblib.dump(feature_cols, "feature_cols.pkl")
```

Create this folder in the project:

```text
model
```

Place the files here:

```text
model/icu_lstm_model.h5
model/scaler.pkl
model/feature_cols.pkl
```

`feature_cols.pkl` should match the corrected model feature list:

```python
[
    "Age", "Gender", "Heart_Rate", "Systolic_BP", "Diastolic_BP",
    "SpO2", "Respiratory_Rate", "Temperature", "WBC", "Lactate",
    "Creatinine", "Platelets", "Hemoglobin", "Glucose",
    "Sodium", "Potassium", "BUN"
]
```

## Run In VS Code

Open the project folder in VS Code:

```text
Mini-Project
```

Terminal 1:

```powershell
node server.js
```

Open:

```text
http://127.0.0.1:8000
```

Terminal 2:

```powershell
python model_api.py
```

If the real model files are not present, the API returns a clinical fallback score so the website does not break. Once the files are added, it automatically uses the LSTM model.
