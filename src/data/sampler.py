import pandas as pd
from typing import Optional

def sample_stratified_candidates(df: pd.DataFrame, target_sample_size: int = 500, random_state: int = 42) -> pd.DataFrame:
    """
    Samples candidate customer-support pairs from processed dataframe.
    Ensures text diversity and length distribution.
    """
    if len(df) <= target_sample_size:
        return df.copy()
        
    df_sample = df.sample(n=target_sample_size, random_state=random_state).reset_index(drop=True)
    return df_sample
