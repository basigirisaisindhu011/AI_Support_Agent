import os
import pandas as pd
from pathlib import Path
from typing import Optional
from src.config import get_config

REQUIRED_COLUMNS = ["tweet_id", "author_id", "in_reply_to_tweet_id", "created_at", "text", "response_tweet_id"]

def load_raw_data(
    file_path: Optional[str] = None,
    max_rows: Optional[int] = None,
    use_sample_if_missing: bool = True
) -> pd.DataFrame:
    """
    Loads raw Customer Support on Twitter CSV.
    Supports max_rows for fast processing and fallback to sample CSV for unit tests.
    """
    config = get_config()
    target_path = Path(file_path) if file_path else config.raw_csv

    if not target_path.exists():
        if use_sample_if_missing and config.sample_csv_path.exists():
            print(f"[Loader] Raw file {target_path} not found. Falling back to sample dataset: {config.sample_csv_path}")
            target_path = config.sample_csv_path
        else:
            raise FileNotFoundError(
                f"Raw dataset not found at {target_path}. "
                "Please download twcs.csv from Kaggle (thoughtvector/customer-support-on-twitter) "
                f"and place it in {config.raw_csv}"
            )

    print(f"[Loader] Loading data from {target_path} (max_rows={max_rows})...")
    
    # Optimize dtypes for low memory footprint
    dtypes = {
        "tweet_id": "str",
        "author_id": "str",
        "in_reply_to_tweet_id": "str",
        "created_at": "str",
        "text": "str",
        "response_tweet_id": "str"
    }

    df = pd.read_csv(
        target_path,
        nrows=max_rows,
        dtype=dtypes,
        usecols=lambda c: c in REQUIRED_COLUMNS,
        low_memory=False
    )
    
    # Ensure missing string values are normalized
    for col in REQUIRED_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()
            df.loc[df[col] == "nan", col] = ""
            df.loc[df[col] == "None", col] = ""

    print(f"[Loader] Loaded {len(df):,} total raw tweets.")
    return df
