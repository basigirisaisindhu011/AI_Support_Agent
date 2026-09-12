import numpy as np
from typing import Dict, Any, List
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_classification_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    """
    Computes intent classification performance metrics.
    Headline metric: Macro F1.
    """
    acc = accuracy_score(y_true, y_pred)
    macro_prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    return {
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(macro_prec), 4),
        "macro_recall": round(float(macro_rec), 4),
        "macro_f1": round(float(macro_f1), 4), # Headline classification metric!
        "weighted_f1": round(float(weighted_f1), 4)
    }

def compute_escalation_trust_metrics(
    gold_should_escalate: List[bool],
    system_auto_handled: List[bool]
) -> Dict[str, float]:
    """
    Computes key escalation trust and safety metrics (Requirement 8):
    1. Unsafe Auto-Handle Rate = (gold_should_escalate AND system_auto_handled) / (all gold_should_escalate)
    2. Automation Rate = auto_handled / all
    3. Escalation Precision, Recall, F1
    """
    total = len(gold_should_escalate)
    if total == 0:
        return {}
        
    gold_esc_count = sum(1 for g in gold_should_escalate if g)
    auto_handled_count = sum(1 for a in system_auto_handled if a)
    
    # Unsafe auto handle: gold required human escalation, but system erroneously auto-handled!
    unsafe_auto_handle_count = sum(
        1 for g, a in zip(gold_should_escalate, system_auto_handled) if g and a
    )
    
    unsafe_auto_handle_rate = unsafe_auto_handle_count / max(1, gold_esc_count)
    automation_rate = auto_handled_count / total
    
    # Binary metrics for predicting should_escalate
    system_escalated = [not a for a in system_auto_handled]
    gold_esc_bool = [bool(g) for g in gold_should_escalate]
    
    esc_prec = precision_score(gold_esc_bool, system_escalated, zero_division=0)
    esc_rec = recall_score(gold_esc_bool, system_escalated, zero_division=0)
    esc_f1 = f1_score(gold_esc_bool, system_escalated, zero_division=0)
    
    return {
        "unsafe_auto_handle_rate": round(float(unsafe_auto_handle_rate), 4),
        "automation_rate": round(float(automation_rate), 4),
        "escalation_precision": round(float(esc_prec), 4),
        "escalation_recall": round(float(esc_rec), 4),
        "escalation_f1": round(float(esc_f1), 4),
        "total_evaluated": total,
        "gold_escalations_total": gold_esc_count,
        "unsafe_auto_handles_count": unsafe_auto_handle_count
    }

def compute_lexical_overlap_secondary_metrics(reference_texts: List[str], generated_texts: List[str]) -> Dict[str, float]:
    """
    Calculates BLEU and ROUGE-L secondary metrics.
    NOTE: Documented as secondary metrics because support responses can be phrased in many ways.
    """
    if not reference_texts or not generated_texts:
        return {"bleu_4": 0.0, "rouge_l": 0.0}
        
    rouge_l_scores = []
    bleu_scores = []
    
    for ref, gen in zip(reference_texts, generated_texts):
        ref_tokens = ref.lower().split()
        gen_tokens = gen.lower().split()
        
        # Simple ROUGE-L approximation (LCS)
        lcs = _longest_common_subsequence(ref_tokens, gen_tokens)
        r_rec = lcs / max(1, len(ref_tokens))
        r_prec = lcs / max(1, len(gen_tokens))
        r_f1 = (2 * r_prec * r_rec) / max(1e-6, (r_prec + r_rec))
        rouge_l_scores.append(r_f1)
        
        # Unigram overlap approximation for BLEU
        overlap = sum(1 for t in gen_tokens if t in ref_tokens)
        bleu = overlap / max(1, len(gen_tokens))
        bleu_scores.append(bleu)
        
    return {
        "bleu_4": round(float(np.mean(bleu_scores)), 4),
        "rouge_l": round(float(np.mean(rouge_l_scores)), 4)
    }

def _longest_common_subsequence(seq1: List[str], seq2: List[str]) -> int:
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]
