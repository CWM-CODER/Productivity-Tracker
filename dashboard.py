
import os

import streamlit as st
import pandas as pd
import sqlite3
import pickle
# ---- Ensure DB exists (Cloud-safe) ----
os.makedirs("db", exist_ok=True)

conn = sqlite3.connect("db/tracker.db")
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

def reset_logs_db():
    conn = sqlite3.connect("db/tracker.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM logs")
    conn.commit()
    conn.close()

# Load trained ML model
import os
import pickle
from ml.semantic_model import ProductivityClassifier
import pandas as pd

MODEL_PATH = "ml/productivity_model.pkl"
BASE_MODEL_PATH = "ml/base_model.pkl"

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        ml_model = pickle.load(f)
else:
    # First run on cloud → train base model
    df = pd.read_csv("ml/productivity_dataset_500.csv")
    texts = df["text"].tolist()
    labels = df["label"].tolist()

    ml_model = ProductivityClassifier()
    ml_model.initial_train(texts, labels)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(ml_model, f)

st.set_page_config(page_title="Productivity Analysis", layout="centered")
st.title("📊 Productivity Tracker Dashboard")

# Load data from DB
conn = sqlite3.connect("db/tracker.db")
df = pd.read_sql_query("SELECT * FROM logs", conn)
conn.close()

if df.empty:
    st.warning("No activity logged yet.")
    st.stop()

# Preprocess
df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0)
df["description"] = df["description"].fillna("")

# ML prediction
df["productive_pred"] = ml_model.predict(df["description"].tolist())
df["productive_conf"] = ml_model.predict_proba(
    df["description"].tolist()
)[:, 1]

df["productive_label"] = df["productive_pred"].map({
    1: "Productive",
    0: "Non-Productive"
})

# Show analysis table
st.subheader("🧠 ML-based Activity Analysis")
st.dataframe(
    df[[
        "date",
        "activity",
        "duration",
        "description",
        "productive_label",
        "productive_conf"
    ]]
)
st.sidebar.header("⚙️ System Controls")
st.sidebar.header("📝 Actions")

if st.sidebar.button("➕ Log New Activity"):
    st.info("Logging new activities is available in local Flask mode.")


if st.sidebar.button("🔄 Factory Reset (Model + Logs)"):
    # 1. Reset model to base
    with open("ml/base_model.pkl", "rb") as f:
        ml_model = pickle.load(f)

    with open("ml/productivity_model.pkl", "wb") as f:
        pickle.dump(ml_model, f)

    # 2. Clear all user logs
    reset_logs_db()

    st.sidebar.success("✅ System reset completed")
    st.rerun()


# Productivity score
productive_time = df[df["productive_pred"] == 1]["duration"].sum()
total_time = df["duration"].sum()

score = int((productive_time / max(total_time, 1)) * 100)
st.metric("📈 Productivity Score", f"{score}/100")

st.caption("Productivity is inferred using NLP (sentence embeddings) + ML classification.")
def save_model(model):
    with open("ml/productivity_model.pkl", "wb") as f:
        pickle.dump(model, f)

st.subheader("🧠 Help Improve the Model")

for i, row in df.iterrows():
    if row["productive_conf"] < 0.8:
        st.write(f"**Activity:** {row['description']}")
        st.write(
            f"Model thinks: **{row['productive_label']}** "
            f"(confidence {row['productive_conf']:.2f})"
        )

        col1, col2 = st.columns(2)

        if col1.button("✅ Productive", key=f"yes_{i}"):
            ml_model.update([row["description"]], [1])
            save_model(ml_model)
            st.success("Model updated ✔")
            st.rerun()

        if col2.button("❌ Not Productive", key=f"no_{i}"):
            ml_model.update([row["description"]], [0])
            save_model(ml_model)
            st.success("Model updated ✔")
            st.rerun()

