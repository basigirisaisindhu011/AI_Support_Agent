import numpy as np
from typing import List, Dict, Any, Optional
from src.retrieval.index import VectorSearchIndex
from src.data.leak_checker import mask_self_retrieval
from src.config import get_config

class HistoricalRetriever:
    """
    Retriever for finding top-k historical support interactions relevant to an incoming query.
    Applies data leakage protection (self-retrieval masking).
    """
    def __init__(self, index: Optional[VectorSearchIndex] = None):
        self.index = index
        self.config = get_config()
        
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        query_tweet_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top_k nearest historical cases for query string.
        Applies self-masking if query_tweet_id is provided.
        """
        if self.index is None or self.index.nn_model is None:
            return []
            
        k = top_k or self.config.top_k
        # Query extra neighbors to account for potential self-masking filtering
        fetch_k = min(k + 5, len(self.index.documents))
        
        if self.index.encoder is not None:
            q_emb = self.index.encoder.encode([query], show_progress_bar=False, convert_to_numpy=True)
        else:
            q_emb = self.index._fallback_vectorizer.transform([query]).toarray()
            
        norm = np.linalg.norm(q_emb)
        if norm > 0:
            q_emb = q_emb / norm
            
        distances, indices = self.index.nn_model.kneighbors(q_emb, n_neighbors=fetch_k)
        distances = distances[0]
        indices = indices[0]
        
        results = []
        for dist, idx in zip(distances, indices):
            doc = dict(self.index.documents[idx])
            sim_score = max(0.0, float(1.0 - dist))
            doc["similarity_score"] = round(sim_score, 4)
            results.append(doc)
            
        # Apply Data Leakage Protection Masking
        if query_tweet_id or query:
            results = mask_self_retrieval(results, query_tweet_id=query_tweet_id or "", query_text=query)
            
        return results[:k]
