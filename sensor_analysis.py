"""
Modul analisis data sensor. Dipakai oleh MCP tools.
Fungsi utama:
- load_data()            : baca sensor_data.txt jadi DataFrame
- correlation_analysis() : matriks korelasi antar sensor (ML: statistik korelasi)
- detect_downtime()      : deteksi kapan tiap sensor "mati" (nilai ~0)
- causal_lag()           : cari lag/downtime antar dua sensor (cross-correlation)
- generate_report()      : rangkum kondisi + rekomendasi
"""
import pandas as pd
import numpy as np

DATA_PATH = "sensor_data.txt"
DEAD_THRESHOLD = 5.0     # nilai di bawah ini dianggap sensor "mati"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


def sensor_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("sensor_")]


def correlation_analysis(df: pd.DataFrame) -> dict:
    """Matriks korelasi Pearson antar semua sensor."""
    cols = sensor_columns(df)
    corr = df[cols].corr(method="pearson").round(3)
    # cari pasangan dengan |korelasi| tertinggi (selain diri sendiri)
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            pairs.append((a, b, float(corr.loc[a, b])))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return {
        "matrix": corr.to_dict(),
        "strongest_pairs": pairs[:3],
    }


def detect_downtime(df: pd.DataFrame) -> dict:
    """Deteksi periode 'mati' tiap sensor (nilai < threshold)."""
    cols = sensor_columns(df)
    result = {}
    for c in cols:
        dead = df[c] < DEAD_THRESHOLD
        # cari blok kontinu yang mati
        events = []
        in_event = False
        start_idx = None
        for i, is_dead in enumerate(dead.values):
            if is_dead and not in_event:
                in_event = True
                start_idx = i
            elif not is_dead and in_event:
                in_event = False
                events.append((start_idx, i - 1))
        if in_event:
            events.append((start_idx, len(dead) - 1))
        result[c] = {
            "total_dead_seconds": int(dead.sum()),
            "num_outages": len(events),
            "outages": [
                {
                    "start": str(df["timestamp"].iloc[s]),
                    "end": str(df["timestamp"].iloc[e]),
                    "duration_sec": int(e - s + 1),
                }
                for s, e in events
            ],
        }
    return result


def causal_lag(df: pd.DataFrame, source: str, target: str, max_lag: int = 20) -> dict:
    """
    Cari downtime berantai: kalau `source` mati, berapa detik kemudian
    `target` ikut mati? Kita bandingkan sinyal 'mati/hidup' (biner) dan
    geser target relatif source untuk cari overlap outage terbaik.
    Ini mendeteksi delay/downtime, bukan sekadar korelasi nilai.
    """
    s_dead = (df[source] < DEAD_THRESHOLD).astype(int).values
    t_dead = (df[target] < DEAD_THRESHOLD).astype(int).values

    # kalau source tidak pernah mati, tak ada efek berantai untuk dianalisis
    if s_dead.sum() == 0:
        return {
            "source": source, "target": target,
            "best_lag_sec": 0, "overlap_score": 0.0,
            "interpretation": f"{source} tidak pernah mati; tak ada efek berantai.",
        }

    best_lag, best_overlap = 0, -1.0
    for lag in range(0, max_lag + 1):
        if lag == 0:
            overlap = np.sum(s_dead & t_dead)
        else:
            overlap = np.sum(s_dead[:-lag] & t_dead[lag:])
        # normalisasi terhadap jumlah 'mati' source
        score = overlap / max(s_dead.sum(), 1)
        if score > best_overlap:
            best_overlap = score
            best_lag = lag

    return {
        "source": source,
        "target": target,
        "best_lag_sec": int(best_lag),
        "overlap_score": round(float(best_overlap), 3),
        "interpretation": (
            f"Ketika {source} mati, {target} ikut mati "
            f"~{best_lag} detik kemudian (downtime berantai, "
            f"overlap {best_overlap:.0%})."
        ),
    }


def generate_report(df: pd.DataFrame) -> dict:
    """Rangkuman lengkap: korelasi + downtime + efek berantai + rekomendasi."""
    corr = correlation_analysis(df)
    downtime = detect_downtime(df)

    # cari sensor yang punya outage sebagai 'source' efek berantai,
    # lalu cek sensor lain mana yang paling ikut mati setelahnya
    cols = sensor_columns(df)
    sources = [c for c in cols if downtime[c]["num_outages"] > 0]
    lag = {"source": None, "target": None, "best_lag_sec": 0,
           "overlap_score": 0.0, "interpretation": "Tidak ada efek berantai terdeteksi."}
    best_score = 0.0
    for src in sources:
        for tgt in cols:
            if tgt == src:
                continue
            cand = causal_lag(df, src, tgt)
            # efek berantai berarti target ikut mati (overlap tinggi) dgn lag > 0
            if cand["overlap_score"] > best_score and cand["best_lag_sec"] >= 0:
                best_score = cand["overlap_score"]
                lag = cand

    # susun rekomendasi berbasis temuan
    recs = []
    for sensor, info in downtime.items():
        if info["num_outages"] > 0:
            recs.append(
                f"{sensor} mengalami {info['num_outages']} kali mati "
                f"(total {info['total_dead_seconds']} detik). "
                f"Periksa catu daya / koneksi {sensor}."
            )
    if lag["source"] and lag["overlap_score"] > 0.5:
        recs.append(
            f"Terdeteksi efek berantai: {lag['source']} → {lag['target']} "
            f"dengan delay {lag['best_lag_sec']} detik "
            f"(overlap {lag['overlap_score']:.0%}). "
            f"Prioritaskan perbaikan {lag['source']} karena kegagalannya "
            f"menyeret {lag['target']} ikut mati {lag['best_lag_sec']} detik kemudian."
        )
    if not recs:
        recs.append("Semua sensor beroperasi normal, tidak ada anomali signifikan.")

    return {
        "correlation": corr,
        "downtime": downtime,
        "chain_effect": lag,
        "recommendations": recs,
    }


if __name__ == "__main__":
    df = load_data()
    import json
    print(json.dumps(generate_report(df), indent=2, default=str))