# app.py
import sys, asyncio
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import os, time, shutil
import streamlit as st
from datetime import datetime
from agent import ChatAgent, MODELS
import store

st.set_page_config(page_title="MCP Chatbot Multi-Sesi", page_icon="🤖")

# File chart yang mungkin dihasilkan tool: (nama_file_sumber, caption)
CHART_SOURCES = [
    ("chart_output.png", "Grafik data sensor"),
    ("forecast_output.png", "Grafik prediksi vs aktual"),
]

@st.cache_resource
def get_loop():
    return asyncio.new_event_loop()

@st.cache_resource
def get_agent():
    a = ChatAgent()
    get_loop().run_until_complete(a.start())
    return a

loop = get_loop()
agent = get_agent()

# ---------- Sidebar ----------
st.sidebar.title("💬 Chat")

new_model = st.sidebar.selectbox("Model for new chat", list(MODELS.keys()))
if st.sidebar.button("➕ New chat"):
    conv = store.new_conversation(new_model)
    st.session_state.active_id = conv["id"]
    st.rerun()

st.sidebar.divider()

convs = store.list_conversations()
by_model = {}
for c in convs:
    by_model.setdefault(c["model_name"], []).append(c)

for model_name, items in by_model.items():
    st.sidebar.caption(f"**{model_name}** ({len(items)})")
    for c in items:
        ts = datetime.fromtimestamp(c["created_at"]).strftime("%d/%m %H:%M")
        col1, col2 = st.sidebar.columns([5, 1])
        if col1.button(f"{c['title']}  ·  {ts}", key=f"open_{c['id']}"):
            st.session_state.active_id = c["id"]
            st.rerun()
        if col2.button("🗑", key=f"del_{c['id']}"):
            store.delete(c["id"])
            if st.session_state.get("active_id") == c["id"]:
                st.session_state.pop("active_id", None)
            st.rerun()

# ---------- Area utama ----------
active_id = st.session_state.get("active_id")

if not active_id:
    st.title("🤖 MCP Chatbot")
    st.info("Create a new chat or choose from sidebar")
    st.stop()

conv = store.load(active_id)

# --- Judul yang bisa diedit ---
edit_key = f"editing_{active_id}"
if st.session_state.get(edit_key):
    col1, col2, col3 = st.columns([6, 1, 1])
    new_title = col1.text_input(
        "Judul", value=conv["title"],
        label_visibility="collapsed", key=f"title_input_{active_id}"
    )
    if col2.button("💾", key=f"save_title_{active_id}"):
        store.rename(active_id, new_title)
        st.session_state[edit_key] = False
        st.rerun()
    if col3.button("✖", key=f"cancel_title_{active_id}"):
        st.session_state[edit_key] = False
        st.rerun()
else:
    col1, col2 = st.columns([8, 1])
    col1.title(f"🤖 {conv['title']}")
    if col2.button("✏️", key=f"edit_title_{active_id}", help="Ubah judul"):
        st.session_state[edit_key] = True
        st.rerun()

st.caption(f"Model: **{conv['model_name']}**")

# Render riwayat pesan (termasuk chart bila pesan punya gambar)
for m in conv["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        for img in m.get("images", []):
            if os.path.exists(img["file"]):
                st.image(img["file"], caption=img.get("caption", ""))

# Input
if prompt := st.chat_input("Tanya sesuatu..."):
    conv["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if conv["title"] == "Percakapan baru":
        conv["title"] = (prompt[:40] + "…") if len(prompt) > 40 else prompt

    with st.chat_message("assistant"):
        # catat waktu modifikasi tiap file chart SEBELUM agent jalan
        before = {
            src: (os.path.getmtime(src) if os.path.exists(src) else 0)
            for src, _ in CHART_SOURCES
        }
        with st.spinner(f"{conv['model_name']} berpikir..."):
            reply = loop.run_until_complete(agent.send(conv, prompt))
        st.markdown(reply)

        # cek tiap sumber chart: kalau baru diperbarui, salin ke nama unik & tampilkan
        new_images = []
        for src, caption in CHART_SOURCES:
            if os.path.exists(src) and os.path.getmtime(src) > before[src]:
                unique = f"{src.rsplit('.', 1)[0]}_{int(time.time()*1000)}.png"
                shutil.copy(src, unique)
                st.image(unique, caption=caption)
                new_images.append({"file": unique, "caption": caption})

    msg = {"role": "assistant", "content": reply}
    if new_images:
        msg["images"] = new_images
    conv["messages"].append(msg)
    store.save(conv)
    st.rerun()