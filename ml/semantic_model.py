from sentence_transformers import SentenceTransformer
from sklearn.linear_model import SGDClassifier

class ProductivityClassifier:
    def __init__(self):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.model = SGDClassifier(loss="log_loss",learning_rate="optimal",alpha=0.001)

    def initial_train(self, texts, labels):
        X = self.embedder.encode(texts)
        self.model.partial_fit(X, labels, classes=[0, 1])

    def update(self, texts, labels):
        X = self.embedder.encode(texts)
        self.model.partial_fit(X, labels)

    def predict(self, texts):
        X = self.embedder.encode(texts)
        return self.model.predict(X)

    def predict_proba(self, texts):
        X = self.embedder.encode(texts)
        return self.model.predict_proba(X)
