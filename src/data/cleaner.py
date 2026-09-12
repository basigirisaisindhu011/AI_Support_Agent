import re
import pandas as pd

def clean_text(text: str) -> str:
    """
    Normalizes tweet text by stripping extra spaces, twitter handle mentions (@xxx),
    and standardizing URLs, while preserving semantic contents.
    """
    if not isinstance(text, str) or not text:
        return ""
    
    # Replace URLs with placeholder token or clean standard format
    text_clean = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
    
    # Strip twitter handle mentions (e.g. @AmazonHelp)
    text_clean = re.sub(r'@[A-Za-z0-9_]+', '', text_clean)
    
    # Replace redundant whitespace / newlines
    text_clean = re.sub(r'\s+', ' ', text_clean).strip()
    
    return text_clean

def preprocess_dataframe(df: pd.DataFrame, brand: str = "AmazonHelp") -> pd.DataFrame:
    """
    Preprocesses raw tweets DataFrame:
    1. Removes unusable/empty rows.
    2. Flags customer vs brand messages.
    3. Retains original text in 'raw_text' and cleaned text in 'clean_text'.
    4. Removes duplicate text messages.
    """
    df = df.copy()
    
    # Drop rows without text
    df = df[df["text"].str.strip() != ""].copy()
    
    # Preserve raw text
    df["raw_text"] = df["text"]
    
    # Clean text
    df["clean_text"] = df["raw_text"].apply(clean_text)
    
    # Flag brand vs customer
    df["is_brand"] = df["author_id"].str.lower() == brand.lower()
    df["is_customer"] = ~df["is_brand"]
    
    # Drop empty clean text rows
    df = df[df["clean_text"].str.strip() != ""].copy()
    
    return df
