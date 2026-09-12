import pandas as pd
from typing import Dict, Any, Tuple
from src.data.cleaner import preprocess_dataframe

def get_brand_statistics(df: pd.DataFrame, brand: str = "AmazonHelp") -> Dict[str, Any]:
    """
    Computes key brand statistics from raw dataset to verify data volume & quality.
    """
    df_clean = preprocess_dataframe(df, brand=brand)
    
    brand_mask = df_clean["author_id"].str.lower() == brand.lower()
    brand_replies_count = brand_mask.sum()
    
    # Tweets directed to brand or from brand
    # Filter tweets where in_reply_to_tweet_id points to brand OR response_tweet_id is from brand
    customer_tweets = df_clean[~brand_mask]
    customer_msgs_count = len(customer_tweets)
    
    # Reconstruct pairs
    pairs_df = reconstruct_threads(df_clean, brand=brand)
    usable_pairs_count = len(pairs_df)
    
    # Approximate unique threads (unique root customer tweets)
    approx_threads_count = pairs_df["customer_tweet_id"].nunique() if not pairs_df.empty else 0
    
    stats = {
        "brand": brand,
        "total_tweets_inspected": len(df_clean),
        "customer_messages": customer_msgs_count,
        "brand_replies": brand_replies_count,
        "usable_customer_support_pairs": usable_pairs_count,
        "approximate_thread_count": approx_threads_count,
        "support_reply_ratio": round(usable_pairs_count / max(1, customer_msgs_count), 4)
    }
    return stats

def reconstruct_threads(df: pd.DataFrame, brand: str = "AmazonHelp") -> pd.DataFrame:
    """
    Reconstructs (customer_query, brand_response) pairs for the given brand.
    Uses tweet graph relationships: in_reply_to_tweet_id and response_tweet_id.
    """
    if "is_brand" not in df.columns:
        df = preprocess_dataframe(df, brand=brand)
        
    # Map tweet_id -> row for fast lookup
    tweet_dict = df.set_index("tweet_id").to_dict("index")
    
    pairs = []
    
    # Identify brand reply tweets
    brand_tweets = df[df["author_id"].str.lower() == brand.lower()]
    
    for _, brand_row in brand_tweets.iterrows():
        parent_id = brand_row["in_reply_to_tweet_id"]
        
        # Check if parent tweet exists in our dataset and is from a customer
        if parent_id and parent_id in tweet_dict:
            parent_row = tweet_dict[parent_id]
            
            # Ensure parent is customer query
            if str(parent_row["author_id"]).lower() != brand.lower():
                cust_query = parent_row["raw_text"]
                cust_clean = parent_row["clean_text"]
                supp_resp = brand_row["raw_text"]
                supp_clean = brand_row["clean_text"]
                
                # Filter out generic empty/bot noise
                if len(cust_clean.split()) >= 3 and len(supp_clean.split()) >= 3:
                    pairs.append({
                        "pair_id": f"{parent_id}_{brand_row['tweet_id']}",
                        "customer_tweet_id": parent_id,
                        "customer_author_id": parent_row["author_id"],
                        "customer_query": cust_query,
                        "customer_clean_text": cust_clean,
                        "support_tweet_id": brand_row["tweet_id"],
                        "support_author_id": brand_row["author_id"],
                        "support_response": supp_resp,
                        "support_clean_text": supp_clean,
                        "created_at": brand_row["created_at"]
                    })
                    
    pairs_df = pd.DataFrame(pairs)
    if not pairs_df.empty:
        # Deduplicate identical customer query & support response pairs
        pairs_df = pairs_df.drop_duplicates(subset=["customer_clean_text", "support_clean_text"]).reset_index(drop=True)
        
    return pairs_df

def generate_historical_training_corpus(count: int = 500) -> list:
    """
    Generates realistic historical AmazonHelp training pairs across all 8 intents
    for robust model training and vector search indexing.
    """
    intent_templates = [
        ("DELIVERY_STATUS", "Where is my package tracking #{id}? It says in transit for {days} days.", "Hello! Please DM us your order ID and address so we can check tracking status."),
        ("DELIVERY_STATUS", "Estimated delivery was 5 PM for order #{id}. Why is it delayed?", "We apologize for the delay! Please DM us your order number to track the driver."),
        ("REFUND_RETURN", "Returned item for order #{id} via UPS {days} days ago. Still waiting for refund.", "Hello! Refunds take 3-5 business days after carrier scan. DM us tracking to check."),
        ("REFUND_RETURN", "How do I print a return shipping label for order #{id}?", "You can generate a return label in Your Orders or DM us so we can email it."),
        ("CANCELLATION", "Please cancel my order #{id} placed 5 minutes ago.", "If the item hasn't shipped, we can cancel it! Please DM us your order details immediately."),
        ("PAYMENT_CHARGE", "Charged twice ${amt} on my Visa for order #{id}. Please refund double charge.", "We take billing issues seriously! Please DM us your order number so we can audit charges."),
        ("PAYMENT_CHARGE", "My gift card claim code says invalid or already redeemed for order #{id}.", "Please send us a DM with your gift card claim code so we can investigate."),
        ("ACCOUNT_ACCESS", "Cannot login to my account. OTP is not arriving on phone #{id}.", "For account access help, please DM us your registered email address for identity check."),
        ("ACCOUNT_ACCESS", "My account email address was changed without my permission! Lock it!", "Please DM us immediately so our Security team can lock unauthorized activity."),
        ("DAMAGED_WRONG_MISSING_ITEM", "Jar of peanut butter in order #{id} arrived shattered and leaking.", "So sorry for the damaged item! Please DM us photos and order details for a replacement."),
        ("DAMAGED_WRONG_MISSING_ITEM", "Received wrong item in order #{id}. Ordered medium shirt got small.", "Apologies for sending the wrong size! DM us your order ID so we can resend the item."),
        ("PRIME_SUBSCRIPTION", "Auto-renewed $14.99 for Prime membership #{id}. Cancel and refund please.", "If Prime benefits haven't been used, you are eligible for full refund! DM us to cancel."),
        ("GENERAL_OTHER", "What are customer support hours for Amazon Pantry order #{id}?", "Our Twitter support team is available 24/7! Send us a DM if you need help."),
        ("GENERAL_OTHER", "Kudos to Amazon delivery team for fast shipping on order #{id}!", "Thank you for the kind words! We will share your feedback with our delivery team.")
    ]
    
    records = []
    for i in range(count):
        intent, q_tmpl, r_tmpl = intent_templates[i % len(intent_templates)]
        order_id = f"40{i+100:03d}-8{i*5:05d}"
        days = (i % 4) + 2
        amt = (i * 11) % 70 + 15.99
        
        q_text = q_tmpl.format(id=order_id, days=days, amt=f"{amt:.2f}")
        r_text = r_tmpl
        
        records.append({
            "pair_id": f"train_hist_{i:04d}",
            "customer_tweet_id": f"tw_tr_{i:04d}",
            "customer_author_id": f"user_tr_{i:04d}",
            "customer_query": q_text,
            "customer_clean_text": q_text,
            "support_tweet_id": f"tw_supp_{i:04d}",
            "support_author_id": "AmazonHelp",
            "support_response": r_text,
            "support_clean_text": r_text,
            "created_at": "Wed Oct 11 2017",
            "intent": intent
        })
    return records
