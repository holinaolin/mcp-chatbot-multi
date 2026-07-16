"""
Generate data sensor dummy dengan pola yang disengaja:
- 4 sensor: A, B, C, D
- Sensor B "bergantung" pada sensor A: kalau A mati (nilai ~0),
  maka B ikut mati beberapa detik kemudian (downtime).
- Sensor C berkorelasi negatif dengan A.
- Sensor D independen (noise), sebagai kontrol.
Output: sensor_data.txt (format CSV dengan header, dipisah koma)
"""
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

N = 2000                      # jumlah baris (detik)
start = datetime(2024, 1, 1, 8, 0, 0)

# --- Sensor A: sinyal dasar dengan beberapa "mati" (nilai jatuh ke ~0) ---
t = np.arange(N)
A = 50 + 20 * np.sin(t / 100) + np.random.normal(0, 2, N)

# Sisipkan beberapa periode "mati" pada A (nilai jatuh mendekati 0)
outage_windows = [(300, 340), (800, 830), (1400, 1470)]  # (mulai, selesai)
for s, e in outage_windows:
    A[s:e] = np.random.normal(1, 0.5, e - s)   # ~0, artinya mati

# --- Sensor B: mengikuti A, tapi telat (downtime) ~5 detik ---
# B mati kalau A mati, dengan jeda 5 detik (efek berantai)
LAG = 5
B = 40 + 0.8 * np.roll(A, LAG) + np.random.normal(0, 2, N)
B[:LAG] = 40 + np.random.normal(0, 2, LAG)
# Pertegas: saat A mati (telat LAG detik), B ikut jatuh ke ~0
for s, e in outage_windows:
    B[s + LAG:e + LAG] = np.random.normal(1, 0.5, (e - s))

# --- Sensor C: korelasi negatif dengan A ---
C = 80 - 0.6 * A + np.random.normal(0, 3, N)

# --- Sensor D: independen (kontrol) ---
D = 30 + 10 * np.sin(t / 37) + np.random.normal(0, 4, N)

# Tulis ke txt (CSV)
with open("sensor_data.txt", "w") as f:
    f.write("timestamp,sensor_A,sensor_B,sensor_C,sensor_D\n")
    for i in range(N):
        ts = (start + timedelta(seconds=i)).strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{ts},{A[i]:.2f},{B[i]:.2f},{C[i]:.2f},{D[i]:.2f}\n")

print(f"Generated sensor_data.txt dengan {N} baris")
print(f"Outage windows pada sensor A (detik): {outage_windows}")
print(f"Sensor B mengikuti A dengan lag {LAG} detik")