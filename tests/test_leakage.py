from src.data.leak_checker import check_golden_set_exclusion, mask_self_retrieval, assert_no_reply_leakage

def test_golden_set_exclusion_assertion():
    train_texts = ["where is my order", "how to return item", "cancel my subscription"]
    golden_texts = ["where is my order"] # Leaked item!
    
    raised = False
    try:
        check_golden_set_exclusion(train_texts, golden_texts)
    except ValueError as exc_info:
        raised = True
        assert "[DATA LEAKAGE ERROR]" in str(exc_info)
    assert raised, "Expected check_golden_set_exclusion to raise ValueError for leaked golden set text."

def test_mask_self_retrieval():
    query_tweet_id = "tw_101"
    query_text = "My package is lost"
    
    retrieved = [
        {"customer_tweet_id": "tw_101", "customer_clean_text": "My package is lost", "support_response": "DM us"}, # Same ID & text
        {"customer_tweet_id": "tw_999", "customer_clean_text": "Item tracking delayed", "support_response": "DM us order number"}
    ]
    
    filtered = mask_self_retrieval(retrieved, query_tweet_id, query_text)
    assert len(filtered) == 1
    assert filtered[0]["customer_tweet_id"] == "tw_999"

def test_assert_no_reply_leakage():
    test_reply = "Please DM us your order ID for refund!"
    retrieved_reply = "Please DM us your order ID for refund!"
    
    is_safe = assert_no_reply_leakage(test_reply, retrieved_reply)
    assert is_safe is False
