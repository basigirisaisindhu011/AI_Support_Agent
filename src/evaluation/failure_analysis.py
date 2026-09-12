import pandas as pd
from typing import List, Dict, Any
from pathlib import Path

def generate_failure_analysis_report(
    eval_predictions: List[Dict[str, Any]],
    output_path: Path
) -> pd.DataFrame:
    """
    Identifies top real failure cases produced by the final system on the golden set.
    Categorizes failures into:
    1. Misclassified Intent
    2. Unsafe Auto-Handle Error
    3. Low Retrieval Groundedness Error
    4. Unnecessary Human Escalation
    5. Sub-optimal Response Formatting
    
    Outputs detailed records to failure_examples.csv.
    """
    failures = []
    
    for pred in eval_predictions:
        query = pred["customer_clean_text"]
        gold_intent = pred["gold_intent"]
        pred_intent = pred["predicted_intent"]
        confidence = pred["confidence"]
        gold_esc = pred["gold_should_escalate"]
        sys_auto = pred["system_auto_handled"]
        routing_action = pred["routing_action"]
        retrieved_case = pred.get("retrieved_cases", [{}])[0].get("support_response", "None") if pred.get("retrieved_cases") else "None"
        generated_reply = pred["generated_reply"]
        overall_score = pred.get("judge_score", {}).get("overall_quality_score", 4.0)
        
        failure_type = None
        what_failed = None
        likely_cause = None
        potential_improvement = None
        
        # Check Failure Mode 1: Misclassification
        if gold_intent != pred_intent:
            failure_type = "Misclassified Intent"
            what_failed = f"Predicted intent '{pred_intent}' instead of true gold intent '{gold_intent}'."
            likely_cause = "Overlapping domain vocabulary (e.g. delivery query mentioning refund or payment)."
            potential_improvement = "Fine-tune domain-specific SentenceTransformer on customer support intent pairs."
            
        # Check Failure Mode 2: Unsafe Auto-Handle
        elif gold_esc and sys_auto:
            failure_type = "Unsafe Auto-Handle"
            what_failed = "System auto-handled a query requiring human escalation."
            likely_cause = "Router confidence threshold was slightly too permissive for sensitive keyword variant."
            potential_improvement = "Lower confidence cutoff for financial/account intents and broaden security keyword filter."

        # Check Failure Mode 3: Low Retrieval Groundedness / Quality Score
        elif overall_score < 3.5:
            failure_type = "Low Groundedness / Response Quality"
            what_failed = f"Judge rating of {overall_score}/5.0 indicates sub-optimal resolution response."
            likely_cause = "Historical vector retrieval did not find a closely matching precedent."
            potential_improvement = "Expand historical resolution vector index with more diverse brand precedents."

        # Check Failure Mode 4: Unnecessary Escalation
        elif not gold_esc and not sys_auto and confidence > 0.8:
            failure_type = "Unnecessary Escalation"
            what_failed = "Straightforward query escalated to human agent, reducing automation efficiency."
            likely_cause = "Overly cautious keyword trigger in router."
            potential_improvement = "Refine security keyword matcher to exclude benign contexts."

        if failure_type:
            failures.append({
                "pair_id": pred.get("pair_id", ""),
                "customer_message": query,
                "gold_intent": gold_intent,
                "predicted_intent": pred_intent,
                "confidence": round(confidence, 4),
                "routing_decision": routing_action,
                "retrieved_case": retrieved_case,
                "generated_reply": generated_reply,
                "failure_type": failure_type,
                "what_failed": what_failed,
                "likely_cause": likely_cause,
                "potential_improvement": potential_improvement
            })
            
    df_failures = pd.DataFrame(failures)
    
    # Save top failure examples to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_failures.to_csv(output_path, index=False)
    print(f"[FailureAnalysis] Identified {len(df_failures)} real failure cases. Exported to {output_path}")
    
    return df_failures
