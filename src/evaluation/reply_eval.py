import numpy as np
from typing import List, Dict, Any
from src.generation.generator import TrivialReplyGenerator, Simple1NNReplyGenerator, MainGroundedReplyGenerator
from src.evaluation.judge import LLMSupportJudge
from src.evaluation.metrics import compute_lexical_overlap_secondary_metrics

def evaluate_all_reply_baselines(
    golden_examples: List[Dict[str, Any]],
    retrieval_func: Any
) -> Dict[str, Dict[str, float]]:
    """
    Evaluates all 3 Reply Generation Systems (Trivial, 1-NN Simple, Grounded Main)
    under the EXACT SAME rubric and Judge.
    """
    generators = {
        "trivial_baseline": TrivialReplyGenerator(),
        "simple_1nn_baseline": Simple1NNReplyGenerator(),
        "main_grounded_system": MainGroundedReplyGenerator()
    }
    
    judge = LLMSupportJudge()
    system_results = {}
    
    for sys_name, gen in generators.items():
        eval_records = []
        gen_texts = []
        gold_texts = []
        
        for item in golden_examples:
            query = item["customer_clean_text"]
            intent = item.get("gold_intent", "GENERAL_OTHER")
            gold_reply = item.get("support_response", "")
            tweet_id = item.get("customer_tweet_id", "")
            
            # Retrieve historical cases with leak protection masking
            retrieved = retrieval_func(query, top_k=3, query_tweet_id=tweet_id)
            
            # Generate reply
            reply = gen.generate_reply(query, intent, retrieved)
            gen_texts.append(reply)
            gold_texts.append(gold_reply)
            
            # Judge evaluation
            judge_res = judge.evaluate_reply(query, intent, reply, retrieved, gold_reply)
            eval_records.append(judge_res)
            
        # Aggregate judge scores
        avg_quality = float(np.mean([r["overall_quality_score"] for r in eval_records]))
        avg_relevance = float(np.mean([r["relevance"] for r in eval_records]))
        avg_groundedness = float(np.mean([r["groundedness"] for r in eval_records]))
        avg_helpfulness = float(np.mean([r["helpfulness"] for r in eval_records]))
        avg_tone = float(np.mean([r["tone"] for r in eval_records]))
        avg_safety = float(np.mean([r["safety"] for r in eval_records]))
        avg_hallucination_risk = float(np.mean([r["hallucination_risk"] for r in eval_records]))
        
        # Secondary lexical overlap metrics
        lexical_metrics = compute_lexical_overlap_secondary_metrics(gold_texts, gen_texts)
        
        system_results[sys_name] = {
            "overall_quality_score": round(avg_quality, 4), # Primary reply evaluation metric!
            "relevance": round(avg_relevance, 4),
            "groundedness": round(avg_groundedness, 4),
            "helpfulness": round(avg_helpfulness, 4),
            "tone": round(avg_tone, 4),
            "safety": round(avg_safety, 4),
            "hallucination_risk": round(avg_hallucination_risk, 4), # 1=low risk, 5=high risk
            "secondary_bleu_4": lexical_metrics["bleu_4"],
            "secondary_rouge_l": lexical_metrics["rouge_l"]
        }
        
    return system_results
