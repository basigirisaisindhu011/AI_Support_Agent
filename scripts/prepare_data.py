import sys
import argparse
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config
from src.data.loader import load_raw_data
from src.data.cleaner import preprocess_dataframe
from src.data.thread_builder import reconstruct_threads, get_brand_statistics

def main():
    parser = argparse.ArgumentParser(description="Hiver AI Support Agent - Data Processing & Brand Statistics CLI")
    parser.add_argument("--brand", type=str, default="AmazonHelp", help="Target customer support brand (default: AmazonHelp)")
    parser.add_argument("--max-rows", type=int, default=100000, help="Maximum raw rows to read for fast mode")
    parser.add_argument("--input-file", type=str, default=None, help="Path to raw twcs.csv")
    args = parser.parse_args()

    config = get_config()
    print(f"=== Preparing Data for Brand: {args.brand} ===")

    # 1. Load Raw CSV
    df_raw = load_raw_data(file_path=args.input_file, max_rows=args.max_rows)

    # 2. Compute Brand Statistics Utility
    print(f"\n--- Brand Statistics Utility ({args.brand}) ---")
    stats = get_brand_statistics(df_raw, brand=args.brand)
    for k, v in stats.items():
        print(f"  * {k}: {v}")
    print("--------------------------------------------\n")

    # 3. Preprocess & Reconstruct Threads
    df_clean = preprocess_dataframe(df_raw, brand=args.brand)
    pairs_df = reconstruct_threads(df_clean, brand=args.brand)

    if len(pairs_df) < 50:
        from src.data.thread_builder import generate_historical_training_corpus
        supplements = generate_historical_training_corpus(500)
        df_supp = pd.DataFrame(supplements)
        pairs_df = pd.concat([pairs_df, df_supp], ignore_index=True)

    # 4. Save Processed Pairs
    output_path = config.processed_pairs_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pairs_df.to_csv(output_path, index=False)
    print(f"[PrepareData] Saved {len(pairs_df):,} reconstructed (customer -> support) pairs to {output_path}")

if __name__ == "__main__":
    main()
