# 🧠 ML-based Productivity Tracker

A smart productivity tracker that uses **NLP sentence embeddings + online machine learning**
to classify activities as productive or non-productive.

## Features
- Flask-based activity logging
- Streamlit analytics dashboard
- Sentence embeddings (MiniLM)
- Online learning via user feedback
- Confidence-based feedback loop
- Factory reset (model + logs)

## Tech Stack
- Python
- Flask
- Streamlit
- scikit-learn (SGDClassifier)
- Sentence Transformers
- SQLite

## How to Run Locally

```bash
pip install -r requirements.txt
python init_db.py
python ml/train_model.py
python app.py
streamlit run dashboard.py
