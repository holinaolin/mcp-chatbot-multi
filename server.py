# server.py
from mcp.server.fastmcp import FastMCP
import json

mcp = FastMCP("demo-tools")

def log(msg: str):
    with open("tool_calls.log", "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# ------------------- Tool lama -------------------

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

# ------------------- Tool sensor (baru) -------------------
# Impor modul analisis. Diletakkan di sini (bukan di atas) agar tool lain
# tetap jalan walau pandas/dll belum terinstal.
import sensor_analysis as sa

@mcp.tool()
def analyze_sensors() -> str:
    """Analisis lengkap semua sensor: korelasi antar sensor, periode downtime,
    efek berantai (sensor mana menyeret sensor lain mati), dan rekomendasi tindakan.
    Gunakan saat user menanyakan kondisi sensor atau minta rekomendasi."""
    log("[analyze_sensors] dipanggil")
    df = sa.load_data()
    return json.dumps(sa.generate_report(df), indent=2, default=str)

@mcp.tool()
def get_correlation() -> str:
    """Matriks korelasi antar sensor + pasangan dengan korelasi terkuat."""
    log("[get_correlation] dipanggil")
    df = sa.load_data()
    return json.dumps(sa.correlation_analysis(df), indent=2, default=str)

@mcp.tool()
def get_downtime() -> str:
    """Daftar periode 'mati' (outage) tiap sensor beserta durasinya dalam detik."""
    log("[get_downtime] dipanggil")
    df = sa.load_data()
    return json.dumps(sa.detect_downtime(df), indent=2, default=str)

@mcp.tool()
def make_chart(window_start: int = 0, window_end: int = 500) -> str:
    """Buat grafik garis nilai semua sensor pada rentang waktu (indeks detik).
    Menyimpan PNG ke file chart_output.png dan mengembalikan nama file itu.
    Pakai saat user minta 'chart', 'grafik', atau 'visualisasi' data sensor."""
    log(f"[make_chart] window={window_start}-{window_end}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = sa.load_data()
    cols = sa.sensor_columns(df)
    sub = df.iloc[window_start:window_end]

    fig, ax = plt.subplots(figsize=(12, 5))
    for c in cols:
        ax.plot(sub["timestamp"], sub[c], label=c, linewidth=1)

    # tandai area outage yang jatuh di rentang ini
    downtime = sa.detect_downtime(df)
    for sensor, info in downtime.items():
        for out in info["outages"]:
            s_ts = sa.pd.to_datetime(out["start"])
            e_ts = sa.pd.to_datetime(out["end"])
            if sub["timestamp"].iloc[0] <= s_ts <= sub["timestamp"].iloc[-1]:
                ax.axvspan(s_ts, e_ts, color="red", alpha=0.1)

    ax.set_title(f"Data Sensor (detik {window_start}-{window_end}) — area merah = outage")
    ax.set_xlabel("waktu"); ax.set_ylabel("nilai"); ax.legend()
    fig.tight_layout()
    fig.savefig("chart_output.png", dpi=80)
    plt.close(fig)
    return "Chart tersimpan sebagai chart_output.png"
# =========================================================================
# TOOLS MACHINE LEARNING — tambahkan blok ini ke server.py yang sudah ada,
# SETELAH tool sensor (analyze_sensors dll) dan SEBELUM baris `if __name__`.
# =========================================================================
import sensor_ml as ml

@mcp.tool()
def forecast_sensor(sensor: str, steps_ahead: int = 10) -> str:
    """Prediksi nilai sebuah sensor beberapa detik ke depan menggunakan model
    Random Forest. Argumen: nama sensor (mis. 'sensor_A') dan berapa detik ke depan.
    Gunakan saat user bertanya 'prediksi', 'ramalkan', atau 'nilai ke depan'."""
    log(f"[forecast_sensor] sensor={sensor} steps={steps_ahead}")
    return json.dumps(ml.forecast_sensor(sensor, steps_ahead), indent=2, default=str)

@mcp.tool()
def detect_anomalies(contamination: float = 0.02) -> str:
    """Deteksi pembacaan sensor yang tidak normal (anomali) menggunakan
    IsolationForest, tanpa threshold manual. Gunakan saat user bertanya soal
    'anomali', 'pembacaan aneh', atau 'ada yang tidak normal?'."""
    log(f"[detect_anomalies] contamination={contamination}")
    return json.dumps(ml.detect_anomalies(contamination), indent=2, default=str)

@mcp.tool()
def predict_failure(sensor: str, horizon: int = 5) -> str:
    """Prediksi apakah sebuah sensor akan MATI dalam beberapa detik ke depan
    menggunakan model klasifikasi (peringatan dini). Argumen: nama sensor dan
    horizon detik. Gunakan saat user bertanya 'apakah sensor akan gagal/mati'."""
    log(f"[predict_failure] sensor={sensor} horizon={horizon}")
    return json.dumps(ml.predict_failure(sensor, horizon), indent=2, default=str)

@mcp.tool()
def forecast_chart(sensor: str, steps_ahead: int = 20) -> str:
    """Buat GRAFIK prediksi vs aktual untuk sebuah sensor dan simpan ke
    forecast_output.png. Grafik menampilkan data aktual, prediksi model pada
    data yang diketahui (untuk menilai akurasi), dan ramalan ke depan.
    Gunakan saat user minta 'grafik prediksi', 'chart forecast', atau
    'tampilkan prediksi sensor secara visual'."""
    log(f"[forecast_chart] sensor={sensor} steps={steps_ahead}")
    return json.dumps(ml.forecast_chart(sensor, steps_ahead), indent=2, default=str)

if __name__ == "__main__":
    mcp.run(transport="stdio")