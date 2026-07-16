"""
code_executor.py — memungkinkan AI menulis & menjalankan kode Python sendiri
untuk analisis/ML dinamis atas data sensor.

Versi ini memakai THREAD (bukan multiprocessing) agar andal di Windows saat
dijalankan dari dalam server MCP. Trade-off: thread yang timeout tidak bisa
dipaksa mati sekeras Process.terminate(); ia jadi daemon dan berhenti saat app
ditutup. Untuk prototipe ini dapat diterima.

================================ PERINGATAN =================================
Prototipe PEMBUKTIAN KONSEP. Batas di sini lapisan DASAR, BUKAN sandbox nyata.
Sebelum produksi WAJIB: sandbox level-OS (container/gVisor/firejail), batas
memori & CPU yang dipaksakan OS, isolasi jaringan, audit trail, validasi hasil
model. Blokir string hanya mencegah kecelakaan, BUKAN serangan.
============================================================================
"""
import io
import threading
import contextlib
import traceback

import pandas as pd
import numpy as np
import data_source

ALLOWED_IMPORTS = {
    "pandas", "numpy", "sklearn", "scipy", "math", "statistics",
    "matplotlib", "json", "datetime", "joblib", "imblearn"
}

BLOCKED_PATTERNS = [
    "import os", "import sys", "import subprocess", "import shutil",
    "open(", "__import__", "eval(", "exec(", "compile(",
    "socket", "requests", "urllib", "pickle", "globals(", "locals(",
    "setattr", "getattr(__", "delattr", "os.", "sys.", "subprocess.",
    "remove(", "rmdir", "unlink", "rmtree", "system(", "popen",
]

MAX_OUTPUT_CHARS = 5000
TIMEOUT_SECONDS = 15


def _screen_code(code: str) -> str | None:
    low = code.lower()
    for pat in BLOCKED_PATTERNS:
        if pat.lower() in low:
            return f"Kode ditolak: mengandung pola terlarang '{pat}'."
    for line in code.splitlines():
        s = line.strip()
        if s.startswith("import ") or s.startswith("from "):
            mod = s.replace("import ", "").replace("from ", "").split()[0].split(".")[0]
            if mod not in ALLOWED_IMPORTS:
                return (f"Kode ditolak: import '{mod}' tidak diizinkan. "
                        f"Yang boleh: {sorted(ALLOWED_IMPORTS)}.")
    return None


def _build_safe_globals():
    df = data_source.get_sensor_data().copy()   # salinan; sumber tak bisa diubah

    real_import = (__builtins__["__import__"] if isinstance(__builtins__, dict)
                   else __builtins__.__import__)

    def guarded_import(name, *args, **kwargs):
        root = name.split(".")[0]
        if root not in ALLOWED_IMPORTS:
            raise ImportError(f"Import '{name}' tidak diizinkan di sandbox.")
        return real_import(name, *args, **kwargs)

    return {
        "__builtins__": {
            "__import__": guarded_import,
            "print": print, "len": len, "range": range, "list": list,
            "dict": dict, "set": set, "tuple": tuple, "str": str,
            "int": int, "float": float, "bool": bool, "abs": abs,
            "min": min, "max": max, "sum": sum, "sorted": sorted,
            "enumerate": enumerate, "zip": zip, "round": round,
            "map": map, "filter": filter, "__build_class__": __build_class__,
            "isinstance": isinstance, "hasattr": hasattr,
        },
        "pd": pd, "np": np, "df": df,
    }


def run_ai_code(code: str) -> dict:
    """Jalankan kode Python yang ditulis AI atas data sensor.
    Data tersedia sebagai DataFrame `df`. pandas=`pd`, numpy=`np`.
    Kode harus mem-`print` hasil yang ingin dilihat."""
    blocked = _screen_code(code)
    if blocked:
        return {"ok": False, "output": blocked, "code": code}

    result = {}

    def target():
        try:
            safe_globals = _build_safe_globals()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                exec(code, safe_globals)
            out = buf.getvalue()
            if len(out) > MAX_OUTPUT_CHARS:
                out = out[:MAX_OUTPUT_CHARS] + "\n...[output dipotong]"
            result["ok"] = True
            result["output"] = out or "(kode jalan tanpa output print)"
        except Exception:
            result["ok"] = False
            result["output"] = traceback.format_exc(limit=3)

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(TIMEOUT_SECONDS)

    if t.is_alive():
        return {"ok": False,
                "output": f"Kode dihentikan: melebihi batas waktu {TIMEOUT_SECONDS} detik.",
                "code": code}

    result["code"] = code
    return result


if __name__ == "__main__":
    print("=== TES 1: ukuran & kolom ===")
    print(run_ai_code("print(df.shape); print(list(df.columns))")["output"])

    print("=== TES 2: kode ML (AI menulis regresi) ===")
    r = run_ai_code(
        "from sklearn.linear_model import LinearRegression\n"
        "X = df[['sensor_A']].values; y = df['sensor_B'].values\n"
        "m = LinearRegression().fit(X, y)\n"
        "print('R2:', round(m.score(X, y), 3))"
    )
    print(r["output"])

    print("=== TES 3: kode berbahaya (ditolak) ===")
    print(run_ai_code("import os")["output"])