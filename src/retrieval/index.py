import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from sklearn.neighbors import NearestNeighbors
from src.config import get_config

class VectorSearchIndex:
    """
    Vector Retrieval Index storing historical customer support pairs (query, resolution).
    Uses Sentence Transformers + NearestNeighbors cosine similarity.
    """
    def __init__(self, embedding_model_name: Optional[str] = None):
        config = get_config()
        self.embedding_model_name = embedding_model_name or config.embedding_model_name
        self.encoder = None
        self._fallback_vectorizer = None
        self.nn_model = None
        self.embeddings = None
        self.documents: List[Dict[str, Any]] = []
        self._init_encoder()
        
    def _init_encoder(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(self.embedding_model_name)
        except Exception as e:
            print(f"[VectorIndex] Warning: SentenceTransformer loading fallback ({e}).")
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._fallback_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

    def build_index(self, pairs_df: pd.DataFrame):
        """
        Builds vector index from processed (customer_query, support_response) pairs.
        """
        if pairs_df.empty:
            raise ValueError("Cannot build index on empty pairs DataFrame.")
            
        print(f"[VectorIndex] Building vector index from {len(pairs_df):,} resolved pairs...")
        self.documents = pairs_df.to_dict("records")
        
        texts = [doc.get("customer_clean_text", doc.get("customer_query", "")) for doc in self.documents]
        
        if self.encoder is not None:
            self.embeddings = self.encoder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        else:
            self.embeddings = self._fallback_vectorizer.fit_transform(texts).toarray()
            
        # Normalize vectors for cosine distance calculation
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        self.embeddings = self.embeddings / norms
        
        self.nn_model = NearestNeighbors(n_neighbors=min(20, len(texts)), metric="cosine")
        self.nn_model.fit(self.embeddings)
        print("[VectorIndex] Vector index built successfully.")

    def save(self, file_path: Path):
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "embeddings": self.embeddings,
                "embedding_model_name": self.embedding_model_name,
                "fallback_vectorizer": self._fallback_vectorizer
            }, f)
        print(f"[VectorIndex] Saved index to {file_path}")

    @classmethod
    def load(cls, file_path: Path) -> 'VectorSearchIndex':
        if not file_path.exists():
            raise FileNotFoundError(f"Index file not found at {file_path}")
            
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            
        index = cls(embedding_model_name=data.get("embedding_model_name"))
        index.documents = data["documents"]
        index.embeddings = data["embeddings"]
        index._fallback_vectorizer = data.get("fallback_vectorizer")
        
        index.nn_model = NearestNeighbors(n_neighbors=min(20, len(index.documents)), metric="cosine")
        index.nn_model.fit(index.embeddings)
        return index
