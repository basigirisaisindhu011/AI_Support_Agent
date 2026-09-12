import sys
import json
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config
from src.retrieval.index import VectorSearchIndex

def main():
    config = get_config()
    print("=== Building Historical Resolution Vector Search Index ===")

    if config.processed_pairs_path.exists():
        df_pairs = pd.read_csv(config.processed_pairs_path)
    else:
        print("[BuildIndex] Processed pairs CSV missing. Using golden set candidates...")
        with open(config.golden_set_path, "r", encoding="utf-8") as f:
            df_pairs = pd.DataFrame(json.load(f))

    index = VectorSearchIndex()
    index.build_index(df_pairs)
    index.save(config.vector_index_path)
    print(f"[BuildIndex] Successfully saved index to {config.vector_index_path}")

if __name__ == "__main__":
    main()
