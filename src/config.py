import os
import yaml
from pathlib import Path
from typing import Dict, Any, List

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"

class Config:
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = str(DEFAULT_CONFIG_PATH)
        
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found at {self.config_path}")
            
        with open(self.config_path, "r", encoding="utf-8") as f:
            self._raw_config = yaml.safe_load(f)
            
        self.root_dir = self.config_path.parent.parent
        
        # System
        self.project_name: str = self._raw_config.get("system", {}).get("project_name", "Hiver AI Support Agent")
        self.brand: str = self._raw_config.get("system", {}).get("brand", "AmazonHelp")
        self.seed: int = int(self._raw_config.get("system", {}).get("seed", 42))
        self.fast_sample_rows: int = int(self._raw_config.get("system", {}).get("fast_sample_rows", 100000))
        
        # Paths
        paths = self._raw_config.get("paths", {})
        self.raw_csv = self.root_dir / paths.get("raw_csv", "data/raw/twcs.csv")
        self.processed_dir = self.root_dir / paths.get("processed_dir", "data/processed")
        self.processed_pairs_path = self.root_dir / paths.get("processed_pairs", "data/processed/twcs_amazonhelp.csv")
        self.golden_set_path = self.root_dir / paths.get("golden_set", "data/golden/golden_set.json")
        self.sample_csv_path = self.root_dir / paths.get("sample_csv", "data/samples/sample_twcs.csv")
        self.vector_index_path = self.root_dir / paths.get("vector_index", "data/processed/retrieval_index.pkl")
        self.reports_dir = self.root_dir / paths.get("reports_dir", "reports")
        self.results_json_path = self.root_dir / paths.get("results_json", "reports/results.json")
        self.predictions_csv_path = self.root_dir / paths.get("predictions_csv", "reports/predictions.csv")
        self.failure_csv_path = self.root_dir / paths.get("failure_csv", "reports/failure_examples.csv")
        
        # Intents
        self.intents: List[str] = self._raw_config.get("intents", {}).get("taxonomy", [
            "DELIVERY_STATUS", "REFUND_RETURN", "CANCELLATION", "PAYMENT_CHARGE",
            "ACCOUNT_ACCESS", "DAMAGED_WRONG_MISSING_ITEM", "PRIME_SUBSCRIPTION", "GENERAL_OTHER"
        ])
        
        # Models
        models = self._raw_config.get("models", {})
        self.embedding_model_name: str = models.get("embedding_model", "all-MiniLM-L6-v2")
        self.classifier_max_iter: int = int(models.get("classifier_max_iter", 1000))
        self.tfidf_max_features: int = int(models.get("tfidf_max_features", 5000))
        
        # Retrieval
        retrieval = self._raw_config.get("retrieval", {})
        self.top_k: int = int(retrieval.get("top_k", 3))
        self.similarity_threshold: float = float(retrieval.get("similarity_threshold", 0.45))
        
        # Routing
        routing = self._raw_config.get("routing", {})
        self.min_confidence_threshold: float = float(routing.get("min_confidence_threshold", 0.65))
        self.min_retrieval_sim_threshold: float = float(routing.get("min_retrieval_sim_threshold", 0.45))
        self.sensitive_intents: List[str] = routing.get("sensitive_intents", ["ACCOUNT_ACCESS", "PAYMENT_CHARGE"])
        self.security_legal_keywords: List[str] = routing.get("security_legal_keywords", [
            "lawyer", "legal", "sue", "court", "hacked", "fraud", "unauthorized", "stolen card"
        ])
        
        # Generation
        generation = self._raw_config.get("generation", {})
        self.max_retrieved_context_len: int = int(generation.get("max_retrieved_context_len", 1200))
        self.temperature: float = float(generation.get("temperature", 0.2))
        
        # Evaluation
        evaluation = self._raw_config.get("evaluation", {})
        self.golden_set_size: int = int(evaluation.get("golden_set_size", 200))
        self.human_eval_sample_size: int = int(evaluation.get("human_eval_sample_size", 35))

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        val = self._raw_config
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

_config_instance = None

def get_config(config_path: str = None) -> Config:
    global _config_instance
    if _config_instance is None or config_path is not None:
        _config_instance = Config(config_path)
    return _config_instance
