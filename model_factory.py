"""
model_factory.py — "Pabrik model" ML plug-and-play.

User cukup menyebut: data mana (sensor), task apa, algoritma apa.
Pabrik merakit fitur, melatih, menyimpan, dan menyediakan model siap pakai.

Konsep:
  TASK    = tujuan prediksi (forecast / outage / anomaly)
  ALGO    = algoritma ML (linear / random_forest / ...)
  Model dilatih sekali, disimpan ke models/, dipakai berulang (plug-and-play).

Menambah algoritma/task baru = cukup tambah entri di katalog, tidak ubah logika.
"""
import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                              IsolationForest)
from sklearn.metrics import (mean_absolute_error, r2_score,
                             precision_score, recall_score, f1_score)

import data_source

MODEL_DIR = "models"
DEAD_THRESHOLD = 5.0
N_LAGS = 10
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================ KATALOG ============================
# Algoritma yang tersedia, dipisah per jenis (regresi vs klasifikasi).
ALGO_CATALOG = {
    "regression": {
        "linear_regression": lambda: LinearRegression(),
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1),
    },
    "classification": {
        "logistic_regression": lambda: LogisticRegression(max_iter=1000,
                                                          class_weight="balanced"),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1),
    },
}

# Task yang tersedia + jenisnya. Ini "menu" untuk user/AI.
TASK_CATALOG = {
    "forecast":  {"type": "regression",
                  "desc": "Prediksi nilai sensor N detik ke depan."},
    "outage":    {"type": "classification",
                  "desc": "Prediksi apakah sensor MATI dalam N detik ke depan."},
    "anomaly":   {"type": "unsupervised",
                  "desc": "Deteksi pembacaan tidak normal (tanpa target)."},
}


def catalog() -> dict:
    """Kembalikan daftar task & algoritma yang tersedia (untuk ditunjukkan ke user)."""
    return {
        "tasks": {k: v["desc"] for k, v in TASK_CATALOG.items()},
        "algorithms": {
            "regression": list(ALGO_CATALOG["regression"].keys()),
            "classification": list(ALGO_CATALOG["classification"].keys()),
        },
        "note": "anomaly tidak butuh algoritma (pakai IsolationForest otomatis).",
    }


# ======================= PERAKITAN FITUR =======================
def _make_lag_features(series, n_lags=N_LAGS):
    X, y = [], []
    for i in range(n_lags, len(series)):
        X.append(series[i - n_lags:i]); y.append(series[i])
    return np.array(X), np.array(y)


def _build_dataset(df, target_sensor, task, horizon):
    """Rakit (X, y) sesuai task. Ini 'otak' yang menerjemahkan task jadi data latih."""
    series = df[target_sensor].values.astype(float)

    if task == "forecast":
        return _make_lag_features(series)

    if task == "outage":
        dead = (series < DEAD_THRESHOLD).astype(int)
        X, y = [], []
        for i in range(N_LAGS, len(series) - horizon):
            X.append(series[i - N_LAGS:i])
            y.append(int(dead[i:i + horizon].any()))
        return np.array(X), np.array(y)

    if task == "anomaly":
        # unsupervised: pakai semua sensor sebagai fitur, tak ada y
        cols = data_source.sensor_columns(df)
        return df[cols].values, None

    raise ValueError(f"Task '{task}' tidak dikenal.")


def _model_key(target_sensor, task, algorithm, horizon):
    return f"{task}_{target_sensor}_{algorithm}_h{horizon}"


# ============================ TRAIN ============================
def train_model(target_sensor: str, task: str,
                algorithm: str = "random_forest", horizon: int = 10) -> dict:
    """Latih model sesuai spesifikasi, simpan, kembalikan metrik.
    Plug-and-play: sekali dilatih, tersimpan untuk dipakai predict_with_model."""
    df = data_source.get_sensor_data()

    if target_sensor not in data_source.sensor_columns(df):
        return {"error": f"Sensor '{target_sensor}' tidak ada. "
                         f"Tersedia: {data_source.sensor_columns(df)}"}
    if task not in TASK_CATALOG:
        return {"error": f"Task '{task}' tidak dikenal. Tersedia: {list(TASK_CATALOG)}"}

    task_type = TASK_CATALOG[task]["type"]
    key = _model_key(target_sensor, task, algorithm, horizon)
    path = os.path.join(MODEL_DIR, f"{key}.pkl")

    # ---- ANOMALY (unsupervised, tak perlu algoritma pilihan) ----
    if task == "anomaly":
        X, _ = _build_dataset(df, target_sensor, task, horizon)
        model = IsolationForest(contamination=0.02, random_state=42)
        model.fit(X)
        joblib.dump({"model": model, "task": task, "type": task_type,
                     "features": "all_sensors"}, path)
        n_anom = int((model.predict(X) == -1).sum())
        return {"status": "trained", "model_key": key, "task": task,
                "algorithm": "isolation_forest",
                "info": f"Model anomali dilatih. {n_anom} anomali di data latih.",
                "saved_to": path}

    # ---- REGRESSION / CLASSIFICATION (butuh algoritma) ----
    if algorithm not in ALGO_CATALOG[task_type]:
        return {"error": f"Algoritma '{algorithm}' tidak cocok untuk task '{task}' "
                        f"({task_type}). Tersedia: {list(ALGO_CATALOG[task_type])}"}

    X, y = _build_dataset(df, target_sensor, task, horizon)
    if task_type == "classification" and y.sum() == 0:
        return {"error": f"Sensor '{target_sensor}' tidak pernah mati; "
                        f"tak ada yang bisa dilatih untuk task outage."}

    split = int(len(X) * 0.8)
    model = ALGO_CATALOG[task_type][algorithm]()
    model.fit(X[:split], y[:split])

    # metrik sesuai jenis task
    y_pred = model.predict(X[split:])
    if task_type == "regression":
        metrics = {"mae": round(float(mean_absolute_error(y[split:], y_pred)), 3),
                   "r2": round(float(r2_score(y[split:], y_pred)), 3)}
    else:
        metrics = {
            "precision": round(float(precision_score(y[split:], y_pred, zero_division=0)), 3),
            "recall": round(float(recall_score(y[split:], y_pred, zero_division=0)), 3),
            "f1": round(float(f1_score(y[split:], y_pred, zero_division=0)), 3),
        }

    joblib.dump({"model": model, "task": task, "type": task_type,
                 "target": target_sensor, "horizon": horizon,
                 "algorithm": algorithm, "n_lags": N_LAGS}, path)

    return {"status": "trained", "model_key": key, "task": task,
            "target": target_sensor, "algorithm": algorithm,
            "horizon": horizon, "metrics": metrics, "saved_to": path}


# ============================ PREDICT ============================
def predict_with_model(model_key: str) -> dict:
    """Muat model tersimpan dan beri prediksi atas DATA TERBARU.
    Ini bagian 'tinggal ambil data' — plug-and-play."""
    path = os.path.join(MODEL_DIR, f"{model_key}.pkl")
    if not os.path.exists(path):
        return {"error": f"Model '{model_key}' belum dilatih. "
                        f"Jalankan train_model dulu."}

    bundle = joblib.load(path)
    df = data_source.get_sensor_data()
    task = bundle["task"]

    if task == "anomaly":
        cols = data_source.sensor_columns(df)
        labels = bundle["model"].predict(df[cols].values)
        n = int((labels == -1).sum())
        return {"model_key": model_key, "task": "anomaly",
                "num_anomalies": n,
                "interpretation": f"{n} anomali terdeteksi di data terkini."}

    # regression / classification: pakai N_LAGS nilai terakhir
    target = bundle["target"]
    series = df[target].values.astype(float)
    window = series[-bundle["n_lags"]:].reshape(1, -1)

    if bundle["type"] == "regression":
        pred = float(bundle["model"].predict(window)[0])
        return {"model_key": model_key, "task": task, "target": target,
                "prediction": round(pred, 2),
                "interpretation": f"Prediksi {target} berikutnya: {pred:.2f}"}
    else:
        pred = int(bundle["model"].predict(window)[0])
        prob = float(bundle["model"].predict_proba(window)[0][1])
        return {"model_key": model_key, "task": task, "target": target,
                "will_happen": bool(pred), "probability": round(prob, 3),
                "interpretation": (f"Kemungkinan {target} mati dalam "
                                   f"{bundle['horizon']} detik: {prob:.0%}")}


def list_trained_models() -> dict:
    """Daftar model yang sudah dilatih & tersimpan."""
    models = [f[:-4] for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]
    return {"trained_models": models, "count": len(models)}


if __name__ == "__main__":
    print("=== KATALOG ===")
    print(json.dumps(catalog(), indent=2))

    print("\n=== TRAIN: outage sensor_A, linear... (harusnya tolak: linear bukan classifier) ===")
    print(json.dumps(train_model("sensor_A", "outage", "linear_regression", 10), indent=2))

    print("\n=== TRAIN: outage sensor_A, random_forest, 10 detik ===")
    r = train_model("sensor_A", "outage", "random_forest", 10)
    print(json.dumps(r, indent=2))

    print("\n=== TRAIN: forecast sensor_B, linear_regression ===")
    print(json.dumps(train_model("sensor_B", "forecast", "linear_regression"), indent=2))

    print("\n=== PREDICT pakai model outage yang tadi ===")
    print(json.dumps(predict_with_model(r["model_key"]), indent=2))

    print("\n=== DAFTAR MODEL TERSIMPAN ===")
    print(json.dumps(list_trained_models(), indent=2))