# store.py
import json, os, time, uuid

STORE_DIR = "conversations"
os.makedirs(STORE_DIR, exist_ok=True)

def _path(conv_id: str) -> str:
    return os.path.join(STORE_DIR, f"{conv_id}.json")

def new_conversation(model_name: str) -> dict:
    conv = {
        "id": "conv_" + uuid.uuid4().hex[:8],
        "title": "New chat",
        "model_name": model_name,
        "created_at": time.time(),
        "messages": [],           # untuk tampilan: [{"role","content"}]
        "provider_history": [],   # untuk model: format asli provider
    }
    save(conv)
    return conv

def save(conv: dict) -> None:
    with open(_path(conv["id"]), "w", encoding="utf-8") as f:
        json.dump(conv, f, ensure_ascii=False, indent=2)

def load(conv_id: str) -> dict:
    with open(_path(conv_id), "r", encoding="utf-8") as f:
        return json.load(f)

def list_conversations() -> list[dict]:
    """Ringkasan semua percakapan, terbaru dulu."""
    items = []
    for fn in os.listdir(STORE_DIR):
        if fn.endswith(".json"):
            try:
                c = load(fn[:-5])
                items.append({
                    "id": c["id"],
                    "title": c["title"],
                    "model_name": c["model_name"],
                    "created_at": c["created_at"],
                })
            except Exception:
                pass
    return sorted(items, key=lambda x: x["created_at"], reverse=True)

def delete(conv_id: str) -> None:
    try:
        os.remove(_path(conv_id))
    except FileNotFoundError:
        pass

def rename(conv_id: str, new_title: str) -> None:
    conv = load(conv_id)
    conv["title"] = new_title.strip() or conv["title"]  # abaikan judul kosong
    save(conv)