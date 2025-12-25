from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class ActivityClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.model = LogisticRegression(max_iter=1000)

    def train(self, texts, labels):
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)

    def predict(self, texts):
        X = self.vectorizer.transform(texts)
        return self.model.predict(X)
