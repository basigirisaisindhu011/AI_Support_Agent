import pandas as pd
from src.data.cleaner import clean_text, preprocess_dataframe
from src.data.thread_builder import get_brand_statistics, reconstruct_threads

def test_clean_text():
    raw = "Hey @AmazonHelp my package at https://t.co/xyz123 was delayed!"
    cleaned = clean_text(raw)
    assert "@AmazonHelp" not in cleaned
    assert "https://" not in cleaned
    assert "[URL]" in cleaned
    assert "package" in cleaned

def test_preprocess_dataframe():
    data = [
        {"tweet_id": "1", "author_id": "cust1", "in_reply_to_tweet_id": "", "created_at": "Wed Oct 11", "text": "Help @AmazonHelp", "response_tweet_id": "2"},
        {"tweet_id": "2", "author_id": "AmazonHelp", "in_reply_to_tweet_id": "1", "created_at": "Wed Oct 11", "text": "Hi cust1, DM us!", "response_tweet_id": ""}
    ]
    df = pd.DataFrame(data)
    df_clean = preprocess_dataframe(df, brand="AmazonHelp")
    assert "raw_text" in df_clean.columns
    assert "clean_text" in df_clean.columns
    assert df_clean["is_brand"].sum() == 1

def test_thread_reconstruction():
    data = [
        {"tweet_id": "10", "author_id": "user123", "in_reply_to_tweet_id": "", "created_at": "Wed Oct 11", "text": "My package tracking has no update", "response_tweet_id": "11"},
        {"tweet_id": "11", "author_id": "AmazonHelp", "in_reply_to_tweet_id": "10", "created_at": "Wed Oct 11", "text": "Please DM us your order ID!", "response_tweet_id": ""}
    ]
    df = pd.DataFrame(data)
    pairs = reconstruct_threads(df, brand="AmazonHelp")
    assert len(pairs) == 1
    assert pairs.iloc[0]["customer_tweet_id"] == "10"
    assert pairs.iloc[0]["support_tweet_id"] == "11"
