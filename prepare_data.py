"""Builds data/icu-data.js for the website from the real time-series dataset and the trained model.

  python prepare_data.py

Reads : data/icu_timeseries_dataset.xlsx, model/icu_lstm_model.h5, model/scaler.pkl, model/feature_cols.pkl
Writes: data/icu-data.js  (last 24 readings per patient, the model's risk score at each reading,
                          and test-set metrics reproduced with the same split as the training notebook)
"""
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, precision_recall_fscore_support, accuracy_score

warnings.filterwarnings("ignore")
import tensorflow as tf

ROOT = Path(__file__).parent
SEQ, STEPS = 10, 24
df = pd.read_excel(ROOT / "data" / "icu_timeseries_dataset.xlsx").sort_values(["Patient_ID", "Time_Step"]).reset_index(drop=True)
sex_text = df.groupby("Patient_ID")["Gender"].first()
df["Gender"] = LabelEncoder().fit_transform(df["Gender"])           # same encoding as the notebook (Female=0, Male=1)
cols = joblib.load(ROOT / "model" / "feature_cols.pkl")
scaler = joblib.load(ROOT / "model" / "scaler.pkl")
model = tf.keras.models.load_model(ROOT / "model" / "icu_lstm_model.h5", compile=False)

# Same patient-level split as the training notebook (so test patients are truly held out).
ps = df.groupby("Patient_ID")["Risk_Label"].max().reset_index()
train_ids, test_ids = train_test_split(ps["Patient_ID"].values, test_size=0.2, random_state=42, stratify=ps["Risk_Label"].values)
test_set = set(test_ids.tolist())

def windows(pid):
    p = df[df.Patient_ID == pid]
    a = scaler.transform(p[cols].values)
    return p, a

# --- metrics on held-out patients -------------------------------------------------
Xs, ys, Xlast = [], [], []
for pid in test_ids:
    p, a = windows(pid); l = p["Risk_Label"].values
    for i in range(len(p) - SEQ):
        Xs.append(a[i:i + SEQ]); ys.append(l[i + SEQ]); Xlast.append(a[i + SEQ - 1])
Xs, ys, Xlast = np.array(Xs), np.array(ys), np.array(Xlast)
prob = model.predict(Xs, verbose=0).ravel(); pred = (prob >= 0.5).astype(int)
p_, r_, f_, s_ = precision_recall_fscore_support(ys, pred, labels=[0, 1])
fpr, tpr, _ = roc_curve(ys, prob); idx = np.unique(np.linspace(0, len(fpr) - 1, 70).astype(int))

# Simple baseline for comparison: logistic regression on the latest reading only (same split).
Xtr, ytr, Xtr_last = [], [], []
for pid in train_ids:
    p, a = windows(pid); l = p["Risk_Label"].values
    for i in range(len(p) - SEQ):
        Xtr_last.append(a[i + SEQ - 1]); ytr.append(l[i + SEQ])
lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(np.array(Xtr_last), np.array(ytr))
auc_lr = roc_auc_score(ys, lr.predict_proba(Xlast)[:, 1])

metrics = {
    "auc": round(float(roc_auc_score(ys, prob)), 4), "accuracy": round(float(accuracy_score(ys, pred)), 4),
    "precision": [round(float(v), 3) for v in p_], "recall": [round(float(v), 3) for v in r_], "f1": [round(float(v), 3) for v in f_], "support": [int(v) for v in s_],
    "cm": confusion_matrix(ys, pred).tolist(), "threshold": 0.5,
    "testPatients": int(len(test_ids)), "trainPatients": int(len(train_ids)), "testSequences": int(len(ys)),
    "roc": [[round(float(fpr[i]), 4), round(float(tpr[i]), 4)] for i in idx],
    "baselineAuc": round(float(auc_lr), 4), "baselineName": "Logistic regression on the latest reading",
}

# --- per-patient data for the website -----------------------------------------------
r2 = lambda v: [round(float(x), 2) for x in v]
patients = []
for pid in sorted(df.Patient_ID.unique()):
    p, a = windows(pid); n = len(p)
    probs = [None] * n                                   # model risk after each reading (needs 10 readings)
    wins = np.array([a[i - SEQ + 1:i + 1] for i in range(SEQ - 1, n)])
    out = model.predict(wins, verbose=0).ravel()
    for k, i in enumerate(range(SEQ - 1, n)): probs[i] = round(float(out[k]), 7)
    last = slice(n - STEPS, n); last10 = slice(n - SEQ, n)
    patients.append({
        "id": int(pid), "age": int(p["Age"].iloc[0]), "sex": "M" if sex_text[pid] == "Male" else "F",
        "adm": str(p["Timestamp"].iloc[0])[:10], "n": int(n), "split": "test" if pid in test_set else "train",
        "hr": r2(p["Heart_Rate"].values[last]), "sp": r2(p["SpO2"].values[last]), "sb": r2(p["Systolic_BP"].values[last]),
        "db": r2(p["Diastolic_BP"].values[last]), "rr": r2(p["Respiratory_Rate"].values[last]), "tp": r2(p["Temperature"].values[last]),
        "wb": r2(p["WBC"].values[last]), "lc": r2(p["Lactate"].values[last]),
        "cr": r2(p["Creatinine"].values[last10]), "pl": r2(p["Platelets"].values[last10]), "hg": r2(p["Hemoglobin"].values[last10]),
        "gl": r2(p["Glucose"].values[last10]), "na": r2(p["Sodium"].values[last10]), "k": r2(p["Potassium"].values[last10]), "bn": r2(p["BUN"].values[last10]),
        "risk": probs[last],
    })

blob = {"seqLen": SEQ, "steps": STEPS, "patients": patients, "metrics": metrics}
(ROOT / "data" / "icu-data.js").write_text("window.ICU_DATA=" + json.dumps(blob, separators=(",", ":")) + ";\n", encoding="utf-8")
print("patients:", len(patients), "| test:", sum(p["split"] == "test" for p in patients))
print("AUC", metrics["auc"], "| baseline AUC", metrics["baselineAuc"], "| size KB", round((ROOT / "data" / "icu-data.js").stat().st_size / 1024))
