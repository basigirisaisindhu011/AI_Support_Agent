import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import cohen_kappa_score
from typing import List, Dict, Any

def compute_human_judge_agreement(human_scores: List[float], judge_scores: List[float]) -> Dict[str, float]:
    """
    Computes agreement statistics between Human ground truth quality ratings and LLM/Rule Judge scores.
    Primary metrics for 1-5 ordinal scales: Spearman correlation & Weighted Cohen's Kappa.
    """
    if len(human_scores) < 2 or len(judge_scores) < 2 or len(human_scores) != len(judge_scores):
        return {
            "spearman_correlation": 0.0,
            "weighted_cohens_kappa": 0.0,
            "exact_agreement": 0.0,
            "agreement_within_1_point": 0.0,
            "pearson_correlation": 0.0,
            "sample_size": len(human_scores)
        }
        
    h_arr = np.array(human_scores)
    j_arr = np.array(judge_scores)
    
    # 1. Spearman Correlation (Primary for ordinal 1-5 scale)
    spearman_corr, _ = spearmanr(h_arr, j_arr)
    spearman_corr = 0.0 if np.isnan(spearman_corr) else float(spearman_corr)
    
    # 2. Pearson Correlation (Secondary reference)
    pearson_corr, _ = pearsonr(h_arr, j_arr)
    pearson_corr = 0.0 if np.isnan(pearson_corr) else float(pearson_corr)
    
    # Discretize to integer bins [1, 2, 3, 4, 5] for Kappa & Exact Agreement
    h_int = np.clip(np.round(h_arr), 1, 5).astype(int)
    j_int = np.clip(np.round(j_arr), 1, 5).astype(int)
    
    # 3. Weighted Cohen's Kappa (Quadratic weights for ordinal distance)
    try:
        weighted_kappa = float(cohen_kappa_score(h_int, j_int, weights="quadratic"))
    except Exception:
        weighted_kappa = 0.0
    if np.isnan(weighted_kappa):
        weighted_kappa = 0.0
        
    # 4. Exact Agreement Percentage
    exact_match_count = np.sum(h_int == j_int)
    exact_agreement = float(exact_match_count / len(h_int))
    
    # 5. Agreement within ±1 point
    within_1_count = np.sum(np.abs(h_int - j_int) <= 1)
    within_1_agreement = float(within_1_count / len(h_int))
    
    return {
        "spearman_correlation": round(spearman_corr, 4),
        "weighted_cohens_kappa": round(weighted_kappa, 4),
        "exact_agreement": round(exact_agreement, 4),
        "agreement_within_1_point": round(within_1_agreement, 4),
        "pearson_correlation": round(pearson_corr, 4),
        "sample_size": len(human_scores)
    }
