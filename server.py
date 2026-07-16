# server.py
from mcp.server.fastmcp import FastMCP
import json

mcp = FastMCP("demo-tools")

def log(msg: str):
    with open("tool_calls.log", "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# ==================== Tool dasar ====================

@mcp.tool()
def calculate(expression: str) -> str:
    """Evaluasi ekspresi matematika sederhana, mis. '3 * (4 + 2)'."""
    log(f"[calculate] expression={expression}")
    import ast, operator as op
    ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
           ast.Div: op.truediv, ast.Pow: op.pow, ast.USub: op.neg}
    def ev(node):
        if isinstance(node, ast.Constant): return node.value
        if isinstance(node, ast.BinOp): return ops[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp): return ops[type(node.op)](ev(node.operand))
        raise ValueError("Ekspresi tidak diizinkan")
    return str(ev(ast.parse(expression, mode="eval").body))

@mcp.tool()
def get_weather(city: str) -> str:
    """Ambil cuaca (dummy) untuk sebuah kota."""
    log(f"[get_weather] city={city}")
    data = {"jakarta": "32°C, cerah berawan", "bandung": "24°C, hujan ringan"}
    return data.get(city.lower(), f"Data cuaca untuk {city} tidak tersedia.")

# ==================== Tool analisis sensor (statistik) ====================
import sensor_analysis as sa

@mcp.tool()
def analyze_sensors() -> str:
    """Analisis lengkap semua sensor: korelasi antar sensor, periode downtime,
    efek berantai (sensor mana menyeret sensor lain mati), dan rekomendasi tindakan."""
    log("[analyze_sensors] dipanggil")
    try:
        df = sa.load_data()
        return json.dumps(sa.generate_report(df), indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": f"Gagal analisis: {e}"})

@mcp.tool()
def get_correlation() -> str:
    """Matriks korelasi antar sensor + pasangan dengan korelasi terkuat."""
    log("[get_correlation] dipanggil")
    try:
        df = sa.load_data()
        return json.dumps(sa.correlation_analysis(df), indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": f"Gagal: {e}"})

@mcp.tool()
def get_downtime() -> str:
    """Daftar periode 'mati' (outage) tiap sensor beserta durasinya dalam detik."""
    log("[get_downtime] dipanggil")
    try:
        df = sa.load_data()
        return json.dumps(sa.detect_downtime(df), indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": f"Gagal: {e}"})

@mcp.tool()
def make_chart(window_start: int = 0, window_end: int = 500) -> str:
    """Buat grafik garis nilai semua sensor pada rentang waktu (indeks detik).
    Menyimpan PNG ke chart_output.png. Pakai saat user minta 'chart'/'grafik'/'visualisasi'."""
    log(f"[make_chart] window={window_start}-{window_end}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        df = sa.load_data()
        cols = sa.sensor_columns(df)
        sub = df.iloc[window_start:window_end]

        fig, ax = plt.subplots(figsize=(12, 5))
        for c in cols:
            ax.plot(sub["timestamp"], sub[c], label=c, linewidth=1)
        downtime = sa.detect_downtime(df)
        for sensor, info in downtime.items():
            for out in info["outages"]:
                s_ts = sa.pd.to_datetime(out["start"]); e_ts = sa.pd.to_datetime(out["end"])
                if sub["timestamp"].iloc[0] <= s_ts <= sub["timestamp"].iloc[-1]:
                    ax.axvspan(s_ts, e_ts, color="red", alpha=0.1)
        ax.set_title(f"Data Sensor (detik {window_start}-{window_end}) — area merah = outage")
        ax.set_xlabel("waktu"); ax.set_ylabel("nilai"); ax.legend()
        fig.tight_layout(); fig.savefig("chart_output.png", dpi=80); plt.close(fig)
        return "Chart tersimpan sebagai chart_output.png"
    except Exception as e:
        return json.dumps({"error": f"Gagal buat chart: {e}"})

# ==================== Tool Machine Learning ====================
import sensor_ml as ml

@mcp.tool()
def forecast_sensor(sensor: str, steps_ahead: int = 10) -> str:
    """Prediksi nilai sebuah sensor beberapa detik ke depan (Random Forest).
    Argumen: nama sensor (mis. 'sensor_A') dan berapa detik ke depan."""
    log(f"[forecast_sensor] sensor={sensor} steps={steps_ahead}")
    try:
        return json.dumps(ml.forecast_sensor(sensor, steps_ahead), indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": f"Gagal forecast: {e}"})

@mcp.tool()
def detect_anomalies(contamination: float = 0.02) -> str:
    """Deteksi pembacaan sensor tidak normal (anomali) via IsolationForest,
    tanpa threshold manual."""
    log(f"[detect_anomalies] contamination={contamination}")
    try:
        return json.dumps(ml.detect_anomalies(contamination), indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": f"Gagal deteksi anomali: {e}"})

@mcp.tool()
def predict_failure(sensor: str, horizon: int = 5) -> str:
    """Prediksi apakah sebuah sensor akan MATI dalam beberapa detik ke depan
    (klasifikasi, peringatan dini). Argumen: nama sensor dan horizon detik."""
    log(f"[predict_failure] sensor={sensor} horizon={horizon}")
    try:
        return json.dumps(ml.predict_failure(sensor, horizon), indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": f"Gagal prediksi kegagalan: {e}"})

@mcp.tool()
def forecast_chart(sensor: str, steps_ahead: int = 20) -> str:
    """Buat GRAFIK prediksi vs aktual untuk sebuah sensor, simpan ke forecast_output.png.
    Gunakan saat user minta 'grafik prediksi' atau 'chart forecast'."""
    log(f"[forecast_chart] sensor={sensor} steps={steps_ahead}")
    try:
        return json.dumps(ml.forecast_chart(sensor, steps_ahead), indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": f"Gagal buat grafik forecast: {e}"})

# ==================== Tool eksekusi kode ML dinamis ====================
import code_executor as ce

@mcp.tool()
def execute_python(code: str) -> str:
    """Jalankan kode Python untuk analisis/ML dinamis atas data sensor.
    Gunakan ketika permintaan user butuh analisis yang TIDAK tersedia di tool lain
    (regresi custom, uji hipotesis, model ML yang kamu rancang sendiri, transformasi data).

    Data sensor tersedia sebagai DataFrame pandas `df`
    (kolom: timestamp, sensor_A, sensor_B, sensor_C, sensor_D).
    pandas=`pd`, numpy=`np`. WAJIB pakai print() untuk menampilkan hasil.
    Library yang boleh: pandas, numpy, sklearn, scipy, math, statistics, matplotlib, json, datetime.

    Contoh:
        from sklearn.linear_model import LinearRegression
        X = df[['sensor_A']].values; y = df['sensor_B'].values
        model = LinearRegression().fit(X, y)
        print('R2:', model.score(X, y))
    """
    log(f"[execute_python] {len(code)} chars")
    result = ce.run_ai_code(code)
    return json.dumps({"berhasil": result["ok"], "output": result["output"]},
                      indent=2, default=str)

if __name__ == "__main__":
    mcp.run(transport="stdio")