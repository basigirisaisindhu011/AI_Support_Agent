import numpy as np
from typing import List, Tuple, Dict, Any
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class MajorityClassBaseline:
    """
    Baseline 0: Trivial classifier predicting the most frequent class in the training set.
    """
    def __init__(self):
        self.majority_class: str = "GENERAL_OTHER"
        self.classes_: np.ndarray = np.array([])
        
    def fit(self, X_train: List[str], y_train: List[str]):
        counter = Counter(y_train)
        if counter:
            self.majority_class = counter.most_common(1)[0][0]
        self.classes_ = np.array(list(counter.keys()))
        return self
        
    def predict(self, X: List[str]) -> List[str]:
        return [self.majority_class] * len(X)
        
    def predict_proba(self, X: List[str]) -> np.ndarray:
        n_samples = len(X)
        n_classes = len(self.classes_)
        probas = np.zeros((n_samples, n_classes))
        if self.majority_class in self.classes_:
            idx = list(self.classes_).index(self.majority_class)
            probas[:, idx] = 1.0
        return probas


class TfIdfBaseline:
    """
    Baseline 1: TF-IDF Vectorizer + Logistic Regression Classifier.
    """
    def __init__(self, max_features: int = 5000, max_iter: int = 1000, random_state: int = 42):
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english', ngram_range=(1, 2))
        self.clf = LogisticRegression(max_iter=max_iter, random_state=random_state, class_weight='balanced')
        self.is_fitted = False
        self.classes_ = np.array([])
        
    def fit(self, X_train: List[str], y_train: List[str]):
        X_vec = self.vectorizer.fit_transform(X_train)
        self.clf.fit(X_vec, y_train)
        self.classes_ = self.clf.classes_
        self.is_fitted = True
        return self
        
    def predict(self, X: List[str]) -> List[str]:
        if not self.is_fitted:
            raise ValueError("TfIdfBaseline model is not fitted.")
        X_vec = self.vectorizer.transform(X)
        return self.clf.predict(X_vec).tolist()
        
    def predict_proba(self, X: List[str]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("TfIdfBaseline model is not fitted.")
        X_vec = self.vectorizer.transform(X)
        return self.clf.predict_proba(X_vec)
