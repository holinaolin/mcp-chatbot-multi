"""
generate_messy_data.py — data sensor dummy yang MENIRU kekacauan data nyata.

Berbeda dari data bersih sebelumnya, ini menyisipkan masalah yang lazim di
sensor industri sungguhan:
  1. NILAI HILANG (NaN)        - sensor gagal kirim data (gap)
  2. PEMBACAAN MACET (stuck)   - sensor beku, nilai sama berulang
  3. LONJAKAN (spike)          - gangguan listrik / noise ekstrem
  4. DRIFT                     - sensor makin tidak akurat seiring waktu
  5. OUTAGE                    - mati total (sudah ada sebelumnya)
  6. DUPLIKAT TIMESTAMP        - masalah logging
  7. TIMESTAMP TAK URUT        - data datang tidak berurutan

Output: sensor_data_messy.txt
Tujuan: MENGUJI apakah tool analisis/ML tahan menghadapi data seperti ini.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(7)

N = 2000
start = datetime(2024, 1, 1, 8, 0, 0)
t = np.arange(N)

# --- Sinyal dasar (mirip sebelumnya) ---
A = 50 + 20 * np.sin(t / 100) + np.random.normal(0, 2, N)
outages = [(300, 340), (900, 930), (1500, 1540)]
for s, e in outages:
    A[s:e] = np.random.normal(1, 0.5, e - s)

B = 40 + 0.8 * np.roll(A, 5) + np.random.normal(0, 2, N)
C = 80 - 0.6 * A + np.random.normal(0, 3, N)
D = 30 + 10 * np.sin(t / 37) + np.random.normal(0, 4, N)

df = pd.DataFrame({
    "timestamp": [start + timedelta(seconds=i) for i in range(N)],
    "sensor_A": A, "sensor_B": B, "sensor_C": C, "sensor_D": D,
})

# ===================== SISIPKAN KEKACAUAN =====================

# 1. NILAI HILANG: sensor_D kadang gagal kirim (~3% baris jadi NaN)
missing_idx = np.random.choice(N, size=int(N * 0.03), replace=False)
df.loc[missing_idx, "sensor_D"] = np.nan

# 2. PEMBACAAN MACET: sensor_B beku di satu nilai selama 50 detik
df.loc[600:650, "sensor_B"] = df.loc[600, "sensor_B"]

# 3. LONJAKAN: sensor_C beberapa spike ekstrem (gangguan)
spike_idx = np.random.choice(N, size=8, replace=False)
df.loc[spike_idx, "sensor_C"] = df.loc[spike_idx, "sensor_C"] * np.random.uniform(5, 10, 8)

# 4. DRIFT: sensor_A perlahan bergeser naik di paruh kedua (kalibrasi meleset)
df.loc[1000:, "sensor_A"] += np.linspace(0, 15, N - 1000)

# 5. DUPLIKAT TIMESTAMP: beberapa baris punya timestamp sama
df.loc[750, "timestamp"] = df.loc[749, "timestamp"]
df.loc[751, "timestamp"] = df.loc[749, "timestamp"]

# 6. TIMESTAMP TAK URUT: tukar beberapa baris
idx = list(df.index)
idx[1200], idx[1205] = idx[1205], idx[1200]
df = df.iloc[idx].reset_index(drop=True)

# 7. NILAI NEGATIF MUSTAHIL: sensor_B beberapa kali negatif (error pembacaan)
neg_idx = np.random.choice(N, size=5, replace=False)
df.loc[neg_idx, "sensor_B"] = -np.random.uniform(10, 50, 5)

# tulis apa adanya (kotor)
df.to_csv("sensor_data_messy.txt", index=False)

print(f"Data kotor dibuat: sensor_data_messy.txt ({N} baris)")
print(f"  - NaN di sensor_D: {df['sensor_D'].isna().sum()} baris")
print(f"  - sensor_B macet: indeks 600-650")
print(f"  - Spike di sensor_C: {len(spike_idx)} titik")
print(f"  - Drift di sensor_A: mulai indeks 1000")
print(f"  - Duplikat timestamp: 3 baris")
print(f"  - Timestamp tak urut & nilai negatif disisipkan")