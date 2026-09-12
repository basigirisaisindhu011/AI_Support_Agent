import sys
import json
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config
from src.intents.taxonomy import suggest_intent_from_keywords
from src.intents.train import train_and_compare_classifiers

def main():
    config = get_config()
    print("=== Training & Evaluating Intent Classifiers ===")

    # Load fixed golden set as test benchmark
    with open(config.golden_set_path, "r", encoding="utf-8") as f:
        golden_set = json.load(f)

    X_test = [g["customer_clean_text"] for g in golden_set]
    y_test = [g["gold_intent"] for g in golden_set]

    # Load training pairs
    if config.processed_pairs_path.exists():
        df_pairs = pd.read_csv(config.processed_pairs_path)
    else:
        df_pairs = pd.DataFrame(golden_set)

    # Prevent leakage: exclude test queries from train set
    golden_query_set = set(q.strip().lower() for q in X_test)
    if "customer_clean_text" in df_pairs.columns:
        train_pairs = df_pairs[~df_pairs["customer_clean_text"].str.strip().str.lower().isin(golden_query_set)].copy()
    else:
        train_pairs = df_pairs.copy()

    if "intent" not in train_pairs.columns:
        train_pairs["intent"] = train_pairs["customer_clean_text"].apply(suggest_intent_from_keywords)

    X_train = train_pairs["customer_clean_text"].tolist()
    y_train = train_pairs["intent"].tolist()

    print(f"[Train] Training set: {len(X_train)} samples | Test set: {len(X_test)} samples (golden set).")
    results, main_clf = train_and_compare_classifiers(X_train, y_train, X_test, y_test)

    print("\n--- Classifier Benchmark Comparison ---")
    for model_name, metrics in results.items():
        print(f"\nModel: {model_name}")
        for k, v in metrics.items():
            print(f"  * {k}: {v}")

    print("\n[Train] Model training and evaluation complete.")

if __name__ == "__main__":
    main()
