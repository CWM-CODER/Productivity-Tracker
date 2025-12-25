import pandas as pd
import pickle
from semantic_model import ProductivityClassifier

print("🚀 train_model.py started")

# Load base dataset
df = pd.read_csv("ml/productivity_dataset_500.csv")

TRAIN_TEXTS = df["text"].tolist()
TRAIN_LABELS = df["label"].tolist()

# Train model
model = ProductivityClassifier()
model.initial_train(TRAIN_TEXTS, TRAIN_LABELS)

# Save base model (never changes)
with open("ml/base_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save user model (will evolve)
with open("ml/productivity_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Base model and user model trained & saved")
