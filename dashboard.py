import os
import sqlite3
import pickle
import streamlit as st
import pandas as pd

from ml.semantic_model import ProductivityClassifier

# ---------------- CONFIG ----------------
MODEL_PATH = "ml/productivity_model.pkl"
BASE_DATASET_PATH = "ml/productivity_dataset_500.csv"
DB_PATH = "db/tracker.db"
CONF_THRESHOLD = 0.8

st.set_page_config(page_title="Productivity Analysis", layout="centered")
st.title("📊 Productivity Tracker Dashboard")

# ---------------- ENSURE DB EXISTS ----------------
os.makedirs("db", exist_ok=True)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT,
    activity TEXT,
    duration INTEGER,
    description TEXT
)
""")
conn.commit()
conn.close()

# ---------------- LOAD / TRAIN MODEL ----------------
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        ml_model = pickle.load(f)
else:
    seed_df = pd.read_csv(BASE_DATASET_PATH)
    texts = seed_df["text"].tolist()
    labels = seed_df["label"].tolist()

    ml_model = ProductivityClassifier()
    ml_model.initial_train(texts, labels)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(ml_model, f)

# ---------------- LOAD LOGS ----------------
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("SELECT * FROM logs", conn)
conn.close()

if df.empty:
    st.info("No activities logged yet. Use Flask locally to add logs.")
    st.stop()

df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0)
df["description"] = df["description"].fillna("")

# ---------------- ML PREDICTIONS ----------------
df["productive_pred"] = ml_model.predict(df["description"].tolist())
df["productive_conf"] = ml_model.predict_proba(
    df["description"].tolist()
)[:, 1]

df["productive_label"] = df["productive_pred"].map({
    1: "Productive",
    0: "Non-Productive"
})

# ---------------- DISPLAY TABLE ----------------
st.subheader("🧠 ML-based Activity Analysis")
st.dataframe(df[[
    "date",
    "activity",
    "duration",
    "description",
    "productive_label",
    "productive_conf"
]])

# ---------------- PRODUCTIVITY SCORE ----------------
productive_time = df[df["productive_pred"] == 1]["duration"].sum()
total_time = df["duration"].sum()
score = int((productive_time / max(total_time, 1)) * 100)

st.metric("📈 Productivity Score", f"{score}/100")

# ---------------- FEEDBACK LEARNING ----------------
st.subheader("🧠 Help Improve the Model")

def save_model(model):
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

for i, row in df.iterrows():
    if row["productive_conf"] < CONF_THRESHOLD:
        st.markdown(f"**Activity:** {row['description']}")
        st.markdown(
            f"Model thinks: **{row['productive_label']}** "
            f"(confidence {row['productive_conf']:.2f})"
        )

        col1, col2 = st.columns(2)

        if col1.button("✅ Productive", key=f"p_{i}"):
            ml_model.update([row["description"]], [1])
            save_model(ml_model)
            st.success("Model updated ✔")
            st.rerun()

        if col2.button("❌ Not Productive", key=f"n_{i}"):
            ml_model.update([row["description"]], [0])
            save_model(ml_model)
            st.success("Model updated ✔")
            st.rerun()

# ---------------- SIDEBAR ----------------
st.sidebar.header("📝 Actions")

if st.sidebar.button("➕ Log New Activity"):
    if os.getenv("STREAMLIT_SERVER_RUNNING"):
        st.info("Logging is available in local Flask mode only.")
    else:
        st.markdown(
            '<meta http-equiv="refresh" content="0; url=http://127.0.0.1:5000/log_form">',
            unsafe_allow_html=True
        )

st.sidebar.header("⚙️ System Controls")

def reset_logs_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM logs")
    conn.commit()
    conn.close()

if st.sidebar.button("🔄 Factory Reset (Model + Logs)"):
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
    reset_logs_db()
    st.sidebar.success("System reset completed")
    st.rerun()

# ---------------- FOOTER ----------------
st.caption(
    "Productivity is inferred using NLP sentence embeddings + ML classification."
)
