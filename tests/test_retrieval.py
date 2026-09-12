import pandas as pd
from src.retrieval.index import VectorSearchIndex
from src.retrieval.retriever import HistoricalRetriever

def test_vector_index_and_retriever():
    data = [
        {"customer_tweet_id": "1", "customer_clean_text": "package tracking delayed", "support_response": "DM us order ID"},
        {"customer_tweet_id": "2", "customer_clean_text": "refund not credited to card", "support_response": "DM us tracking number"}
    ]
    df = pd.DataFrame(data)
    index = VectorSearchIndex()
    index.build_index(df)
    
    retriever = HistoricalRetriever(index=index)
    results = retriever.retrieve("tracking is late", top_k=1, query_tweet_id="10")
    
    assert len(results) == 1
    assert "similarity_score" in results[0]
    assert results[0]["similarity_score"] >= 0.0
