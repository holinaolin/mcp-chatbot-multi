"""
Modul Machine Learning untuk data sensor (versi dengan model persisten).

Model dilatih SEKALI lalu disimpan ke folder models/ dengan joblib.
Panggilan berikutnya memuat model dari disk (cepat), tidak melatih ulang.

Kapabilitas:
  1. forecast_sensor()      - prediksi nilai ke depan (RandomForestRegressor)
  2. detect_anomalies()     - anomali tanpa threshold (IsolationForest)
  3. predict_failure()      - prediksi sensor akan mati (RandomForestClassifier)
  4. forecast_chart()       - grafik prediksi vs aktual, simpan PNG
  5. retrain_all()          - hapus & latih ulang semua model (kalau data berubah)
"""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.metrics import (mean_absolute_error, precision_score,
                             recall_score, f1_score)

DATA_PATH = "sensor_data.txt"
MODEL_DIR = "models"
DEAD_THRESHOLD = 5.0
N_LAGS = 10

os.makedirs(MODEL_DIR, exist_ok=True)


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["timestamp"])


def sensor_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("sensor_")]


def _model_path(name: str) -> str:
    return os.path.join(MODEL_DIR, f"{name}.pkl")


def _make_lag_features(series: np.ndarray, n_lags: int = N_LAGS):
    X, y = [], []
    for i in range(n_lags, len(series)):
        X.append(series[i - n_lags:i])
        y.append(series[i])
    return np.array(X), np.array(y)


# ---------------------------------------------------------------------------
# 1. FORECASTING (dengan cache model)
# ---------------------------------------------------------------------------
def _get_forecast_model(sensor: str):
    """Muat model forecast dari disk; latih & simpan kalau belum ada.
    Mengembalikan (model, mae, series)."""
    path = _model_path(f"forecast_{sensor}")
    df = load_data()
    series = df[sensor].values.astype(float)

    if os.path.exists(path):
        bundle = joblib.load(path)
        return bundle["model"], bundle["mae"], series

    X, y = _make_lag_features(series)
    split = int(len(X) * 0.8)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X[:split], y[:split])
    mae = float(mean_absolute_error(y[split:], model.predict(X[split:])))
    joblib.dump({"model": model, "mae": mae}, path)
    return model, mae, series


def forecast_sensor(sensor: str, steps_ahead: int = 10) -> dict:
    df = load_data()
    if sensor not in sensor_columns(df):
        return {"error": f"Sensor '{sensor}' tidak ditemukan."}

    cached = os.path.exists(_model_path(f"forecast_{sensor}"))
    model, mae, series = _get_forecast_model(sensor)

    window = series[-N_LAGS:].tolist()
    preds = []
    for _ in range(steps_ahead):
        p = float(model.predict([window[-N_LAGS:]])[0])
        preds.append(round(p, 2))
        window.append(p)

    return {
        "sensor": sensor,
        "steps_ahead": steps_ahead,
        "predictions": preds,
        "model_mae": round(mae, 3),
        "last_actual": round(float(series[-1]), 2),
        "loaded_from_cache": cached,
        "interpretation": (
            f"Prediksi {sensor} untuk {steps_ahead} detik ke depan. "
            f"Rata-rata error model (MAE) = {mae:.2f} unit."
        ),
    }


# ---------------------------------------------------------------------------
# 2. ANOMALY DETECTION (dengan cache model)
# ---------------------------------------------------------------------------
def _get_anomaly_model(contamination: float, X: np.ndarray):
    path = _model_path(f"anomaly_c{contamination}")
    if os.path.exists(path):
        return joblib.load(path)
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(X)
    joblib.dump(model, path)
    return model


def detect_anomalies(contamination: float = 0.02) -> dict:
    df = load_data()
    cols = sensor_columns(df)
    X = df[cols].values

    model = _get_anomaly_model(contamination, X)
    labels = model.predict(X)
    scores = model.score_samples(X)

    idx = np.where(labels == -1)[0]
    anomalies = [{
        "timestamp": str(df["timestamp"].iloc[i]),
        "values": {c: round(float(df[c].iloc[i]), 2) for c in cols},
        "anomaly_score": round(float(scores[i]), 3),
    } for i in idx]
    anomalies.sort(key=lambda a: a["anomaly_score"])

    return {
        "total_points": len(df),
        "num_anomalies": len(idx),
        "contamination": contamination,
        "top_anomalies": anomalies[:10],
        "interpretation": (
            f"Ditemukan {len(idx)} anomali dari {len(df)} titik data "
            f"menggunakan IsolationForest (tanpa threshold manual)."
        ),
    }


# ---------------------------------------------------------------------------
# 3. FAILURE PREDICTION (dengan cache model)
# ---------------------------------------------------------------------------
def _get_failure_model(sensor: str, horizon: int):
    path = _model_path(f"failure_{sensor}_h{horizon}")
    df = load_data()
    series = df[sensor].values.astype(float)
    dead = (series < DEAD_THRESHOLD).astype(int)

    if os.path.exists(path):
        bundle = joblib.load(path)
        return bundle["model"], bundle["metrics"], series

    X, y = [], []
    for i in range(N_LAGS, len(series) - horizon):
        X.append(series[i - N_LAGS:i])
        y.append(int(dead[i:i + horizon].any()))
    X, y = np.array(X), np.array(y)

    if y.sum() == 0:
        return None, None, series

    split = int(len(X) * 0.8)
    model = RandomForestClassifier(n_estimators=100, random_state=42,
                                   class_weight="balanced", n_jobs=-1)
    model.fit(X[:split], y[:split])
    y_pred = model.predict(X[split:])
    metrics = {
        "precision": round(float(precision_score(y[split:], y_pred, zero_division=0)), 3),
        "recall": round(float(recall_score(y[split:], y_pred, zero_division=0)), 3),
        "f1": round(float(f1_score(y[split:], y_pred, zero_division=0)), 3),
    }
    joblib.dump({"model": model, "metrics": metrics}, path)
    return model, metrics, series


def predict_failure(sensor: str, horizon: int = 5) -> dict:
    df = load_data()
    if sensor not in sensor_columns(df):
        return {"error": f"Sensor '{sensor}' tidak ditemukan."}

    model, metrics, series = _get_failure_model(sensor, horizon)
    if model is None:
        return {"sensor": sensor,
                "interpretation": f"{sensor} tidak pernah mati; tak ada yang diprediksi."}

    current = series[-N_LAGS:].reshape(1, -1)
    will_fail = bool(model.predict(current)[0])
    prob = float(model.predict_proba(current)[0][1])

    return {
        "sensor": sensor,
        "horizon_sec": horizon,
        "will_fail_soon": will_fail,
        "failure_probability": round(prob, 3),
        "model_precision": metrics["precision"],
        "model_recall": metrics["recall"],
        "model_f1": metrics["f1"],
        "interpretation": (
            f"Kemungkinan {sensor} mati dalam {horizon} detik ke depan: {prob:.0%}. "
            f"(Model F1={metrics['f1']:.2f})"
        ),
    }


# ---------------------------------------------------------------------------
# 4. FORECAST CHART — grafik prediksi vs aktual
# ---------------------------------------------------------------------------
def forecast_chart(sensor: str, steps_ahead: int = 20,
                   out_path: str = "forecast_output.png") -> dict:
    df = load_data()
    if sensor not in sensor_columns(df):
        return {"error": f"Sensor '{sensor}' tidak ditemukan."}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model, mae, series = _get_forecast_model(sensor)

    # Bagian A: prediksi pada 100 titik terakhir yang DIKETAHUI (aktual vs prediksi)
    test_len = min(100, len(series) - N_LAGS)
    actual = series[-test_len:]
    predicted = []
    for i in range(len(series) - test_len, len(series)):
        win = series[i - N_LAGS:i]
        predicted.append(float(model.predict([win])[0]))
    predicted = np.array(predicted)

    # Bagian B: ramalan murni ke depan
    window = series[-N_LAGS:].tolist()
    future = []
    for _ in range(steps_ahead):
        p = float(model.predict([window[-N_LAGS:]])[0])
        future.append(p)
        window.append(p)

    fig, ax = plt.subplots(figsize=(12, 5))
    x_actual = np.arange(test_len)
    ax.plot(x_actual, actual, label="Aktual", color="#2563eb", linewidth=1.5)
    ax.plot(x_actual, predicted, label="Prediksi (data diketahui)",
            color="#f59e0b", linestyle="--", linewidth=1.5)
    x_future = np.arange(test_len, test_len + steps_ahead)
    ax.plot(x_future, future, label="Ramalan ke depan",
            color="#dc2626", marker="o", markersize=3, linewidth=1.5)
    ax.axvline(test_len - 1, color="gray", linestyle=":", alpha=0.7)

    ax.set_title(f"Forecast {sensor} — Aktual vs Prediksi (MAE={mae:.2f})")
    ax.set_xlabel("langkah waktu (detik)")
    ax.set_ylabel("nilai")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=80)
    plt.close(fig)

    return {
        "sensor": sensor,
        "chart_saved": out_path,
        "model_mae": round(mae, 3),
        "future_predictions": [round(f, 2) for f in future],
        "interpretation": (
            f"Grafik forecast {sensor} disimpan. Biru=aktual, "
            f"oranye putus-putus=prediksi pada data diketahui (makin dekat makin akurat), "
            f"merah=ramalan {steps_ahead} detik ke depan."
        ),
    }


# ---------------------------------------------------------------------------
# 5. RETRAIN — hapus semua model tersimpan (panggil kalau data berganti)
# ---------------------------------------------------------------------------
def retrain_all() -> dict:
    removed = 0
    for f in os.listdir(MODEL_DIR):
        if f.endswith(".pkl"):
            os.remove(os.path.join(MODEL_DIR, f))
            removed += 1
    return {"removed_models": removed,
            "message": "Semua model dihapus. Akan dilatih ulang saat tool dipanggil lagi."}


if __name__ == "__main__":
    import json, time
    print("=== Panggilan PERTAMA (melatih & menyimpan) ===")
    retrain_all()  # bersihkan dulu supaya benar-benar melatih
    t0 = time.time()
    forecast_sensor("sensor_A", 5)
    predict_failure("sensor_A", 5)
    detect_anomalies()
    print(f"Waktu: {time.time()-t0:.2f} detik")

    print("\n=== Panggilan KEDUA (memuat dari cache) ===")
    t0 = time.time()
    r = forecast_sensor("sensor_A", 5)
    predict_failure("sensor_A", 5)
    detect_anomalies()
    print(f"Waktu: {time.time()-t0:.2f} detik | dari cache: {r['loaded_from_cache']}")

    print("\n=== forecast_chart ===")
    print(json.dumps(forecast_chart("sensor_A"), indent=2, default=str))