import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from sklearn.linear_model import LogisticRegression
from src.config import get_config

class IntentClassifier:
    """
    Main Classifier: MiniLM Sentence Transformer Embeddings + Calibrated Classifier.
    Computes intent prediction and soft confidence score P(intent | query).
    """
    def __init__(self, model_name: Optional[str] = None, random_state: int = 42):
        config = get_config()
        self.model_name = model_name or config.embedding_model_name
        self.random_state = random_state
        self.clf = LogisticRegression(max_iter=1000, random_state=random_state, class_weight='balanced')
        self.encoder = None
        self._fallback_vectorizer = None
        self.is_fitted = False
        self.classes_ = np.array([])
        self._init_encoder()
        
    def _init_encoder(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(self.model_name)
            print(f"[IntentClassifier] Loaded SentenceTransformer model: {self.model_name}")
        except Exception as e:
            print(f"[IntentClassifier] SentenceTransformer warning ({e}). Using TF-IDF embedding fallback.")
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._fallback_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))

    def _encode(self, texts: List[str]) -> np.ndarray:
        if self.encoder is not None:
            embeddings = self.encoder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            return embeddings
        else:
            if not self.is_fitted:
                return self._fallback_vectorizer.fit_transform(texts).toarray()
            else:
                return self._fallback_vectorizer.transform(texts).toarray()

    def fit(self, X_train: List[str], y_train: List[str]):
        if self.encoder is None and self._fallback_vectorizer is not None:
            X_emb = self._fallback_vectorizer.fit_transform(X_train).toarray()
        else:
            X_emb = self._encode(X_train)
            
        self.clf.fit(X_emb, y_train)
        self.classes_ = self.clf.classes_
        self.is_fitted = True
        return self

    def predict(self, X: List[str]) -> List[str]:
        if not self.is_fitted:
            raise ValueError("IntentClassifier is not fitted yet.")
        X_emb = self._encode(X)
        return self.clf.predict(X_emb).tolist()

    def predict_proba(self, X: List[str]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("IntentClassifier is not fitted yet.")
        X_emb = self._encode(X)
        return self.clf.predict_proba(X_emb)

    def predict_with_confidence(self, query: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Returns (predicted_intent, confidence_score, all_class_probabilities).
        """
        probas = self.predict_proba([query])[0]
        max_idx = int(np.argmax(probas))
        pred_intent = str(self.classes_[max_idx])
        confidence = float(probas[max_idx])
        
        all_probas = {str(cls): float(prob) for cls, prob in zip(self.classes_, probas)}
        return pred_intent, confidence, all_probas
