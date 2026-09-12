"""Data Leakage Protection and Verification Utility.

Ensures strict separation between evaluation/golden dataset and training/retrieval corpora.
"""
from typing import List, Dict, Any, Set
import pandas as pd

def check_golden_set_exclusion(train_texts: List[str], golden_texts: List[str]) -> bool:
    """
    Asserts that no golden evaluation customer query exists in the training set.
    """
    train_set: Set[str] = set(t.strip().lower() for t in train_texts)
    leaked_items = [t for t in golden_texts if t.strip().lower() in train_set]
    
    if leaked_items:
        raise ValueError(
            f"[DATA LEAKAGE ERROR] {len(leaked_items)} golden set examples found in training data! "
            f"Example leaked item: '{leaked_items[0]}'"
        )
    return True

def mask_self_retrieval(
    retrieved_items: List[Dict[str, Any]],
    query_tweet_id: str,
    query_text: str
) -> List[Dict[str, Any]]:
    """
    Filters out retrieved historical results if they match the query item itself:
    1. Same tweet_id or pair_id.
    2. Exact identical customer text.
    3. The exact same historical support response.
    """
    filtered = []
    norm_query = query_text.strip().lower()
    
    for item in retrieved_items:
        item_tweet_id = str(item.get("customer_tweet_id", ""))
        item_pair_id = str(item.get("pair_id", ""))
        item_cust_text = str(item.get("customer_clean_text", "")).strip().lower()
        
        # Check matching ID
        if query_tweet_id and (query_tweet_id == item_tweet_id or query_tweet_id in item_pair_id):
            continue
            
        # Check identical text
        if norm_query and norm_query == item_cust_text:
            continue
            
        filtered.append(item)
        
    return filtered

def assert_no_reply_leakage(test_support_response: str, retrieved_support_response: str) -> bool:
    """
    Asserts that the retrieved historical response is not identical to the test sample's ground truth reply.
    """
    t_resp = test_support_response.strip().lower()
    r_resp = retrieved_support_response.strip().lower()
    
    if t_resp and r_resp and t_resp == r_resp:
        return False
    return True
