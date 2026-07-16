"""
data_source.py — SATU PINTU untuk semua akses data sensor.

Seluruh modul (sensor_analysis, sensor_ml) HARUS mengambil data lewat sini,
tidak boleh membaca file/DB langsung. Tujuannya: saat pindah dari file ke
database, cukup ubah SATU fungsi di sini tanpa menyentuh tool atau analisis.

Ganti nanti:
    - Untuk DB, ubah isi _read_raw() jadi query SQL (contoh disertakan).
    - Sisa sistem tidak perlu berubah sama sekali.
"""
import os
import pandas as pd

# Konfigurasi sumber data. Nanti bisa diganti lewat env var / config.
DATA_SOURCE = os.environ.get("SENSOR_DATA_SOURCE", "file")
DATA_FILE = os.environ.get("SENSOR_DATA_FILE", "sensor_data.txt")


def _read_raw() -> pd.DataFrame:
    """Baca data MENTAH dari sumber. HANYA fungsi ini yang tahu asal data.

    Saat pindah ke database, ganti isi fungsi ini, misalnya:

        from sqlalchemy import create_engine
        engine = create_engine(os.environ["DB_URL"])
        return pd.read_sql(
            "SELECT timestamp, sensor_A, sensor_B, sensor_C, sensor_D "
            "FROM sensor_readings ORDER BY timestamp",
            engine, parse_dates=["timestamp"],
        )

    Sisa sistem tidak perlu tahu perubahan ini.
    """
    if DATA_SOURCE == "file":
        if not os.path.exists(DATA_FILE):
            raise FileNotFoundError(
                f"File data '{DATA_FILE}' tidak ditemukan. "
                f"Jalankan generate_data.py atau set SENSOR_DATA_FILE."
            )
        return pd.read_csv(DATA_FILE, parse_dates=["timestamp"])

    # Placeholder untuk sumber lain di masa depan
    raise ValueError(f"Sumber data '{DATA_SOURCE}' belum didukung.")


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Pembersihan dasar yang berlaku untuk SEMUA data, dari mana pun asalnya.
    Di sinilah tempat menangani kekacauan data nyata nanti (nilai hilang, dsb).
    """
    # pastikan ada kolom timestamp
    if "timestamp" not in df.columns:
        raise ValueError("Data tidak punya kolom 'timestamp'.")

    # urutkan berdasarkan waktu (data DB belum tentu terurut)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # buang baris yang timestamp-nya kosong
    df = df.dropna(subset=["timestamp"])

    return df


def get_sensor_data() -> pd.DataFrame:
    """PINTU UTAMA. Semua modul memanggil ini untuk mendapat data sensor
    yang sudah dibaca dan dibersihkan seperlunya."""
    df = _read_raw()
    df = _clean(df)
    return df


def sensor_columns(df: pd.DataFrame) -> list[str]:
    """Daftar kolom sensor (berawalan 'sensor_'). Satu definisi dipakai semua."""
    return [c for c in df.columns if c.startswith("sensor_")]


def data_summary() -> dict:
    """Ringkasan cepat sumber data yang berguna untuk diagnostik & sanity check."""
    df = get_sensor_data()
    cols = sensor_columns(df)
    return {
        "source": DATA_SOURCE,
        "rows": len(df),
        "sensors": cols,
        "time_range": {
            "start": str(df["timestamp"].min()),
            "end": str(df["timestamp"].max()),
        },
        "missing_values": {c: int(df[c].isna().sum()) for c in cols},
    }


if __name__ == "__main__":
    import json
    print(json.dumps(data_summary(), indent=2, default=str))