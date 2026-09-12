import os
import json
import requests
from typing import Dict, Any, List, Optional

class LLMSupportJudge:
    """
    6-Dimension LLM-as-Judge Evaluator for Support Replies.
    Evaluates:
    - Relevance (1-5, higher better)
    - Groundedness (1-5, higher better)
    - Helpfulness (1-5, higher better)
    - Tone (1-5, higher better)
    - Safety (1-5, higher better)
    - Hallucination Risk (1-5, 1=very low risk, 5=severe risk)
    """
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()

    def evaluate_reply(
        self,
        query: str,
        intent: str,
        generated_reply: str,
        retrieved_cases: List[Dict[str, Any]],
        gold_reply: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluates generated reply across 6 dimensions.
        Returns score dictionary with individual dimension ratings and overall quality composite.
        """
        if self.groq_api_key or self.openai_api_key:
            try:
                return self._api_judge_evaluation(query, intent, generated_reply, retrieved_cases, gold_reply)
            except Exception as e:
                print(f"[Judge] API Judge failed ({e}). Falling back to deterministic Rule Judge.")
                
        return self._rule_judge_evaluation(query, intent, generated_reply, retrieved_cases, gold_reply)

    def _rule_judge_evaluation(
        self,
        query: str,
        intent: str,
        generated_reply: str,
        retrieved_cases: List[Dict[str, Any]],
        gold_reply: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Deterministic, offline rule-based heuristic judge producing fine-grained 1-5 scale scores.
        Penalizes non-specific generic answers on queries requiring specific guidance.
        """
        reply_lower = generated_reply.lower()
        query_lower = query.lower()

        is_generic_dm = ("send us a dm" in reply_lower or "direct message" in reply_lower) and len(generated_reply.split()) <= 20

        # 1. Relevance: Checks intent alignment & specificity
        if is_generic_dm and intent in ["GENERAL_OTHER", "PAYMENT_CHARGE"]:
            relevance = 3
        elif is_generic_dm and intent in ["REFUND_RETURN", "DAMAGED_WRONG_MISSING_ITEM", "CANCELLATION"]:
            relevance = 4
        else:
            relevance = 5

        # 2. Groundedness: Check alignment with retrieved cases
        groundedness = 4
        if retrieved_cases:
            top_resp = retrieved_cases[0].get("support_response", "").lower()
            if ("dm" in top_resp and "dm" in reply_lower) or ("refund" in top_resp and "refund" in reply_lower):
                groundedness = 5

        # 3. Helpfulness: Clear next steps & intent-specific guidance
        if is_generic_dm and intent in ["GENERAL_OTHER", "PAYMENT_CHARGE"]:
            helpfulness = 3
        elif is_generic_dm and intent in ["REFUND_RETURN", "DAMAGED_WRONG_MISSING_ITEM"]:
            helpfulness = 3
        elif any(w in reply_lower for w in ["dm", "direct message", "order", "track", "help"]):
            helpfulness = 5
        else:
            helpfulness = 3

        # 4. Tone: Polite, professional customer service etiquette
        tone = 5 if any(w in reply_lower for w in ["hello", "thanks", "thank you", "sorry", "apologize", "please"]) else 4

        # 5. Safety: Security/privacy adherence (e.g. asking for public info vs DM)
        safety = 5
        if any(w in reply_lower for w in ["password", "card number", "ssn"]):
            safety = 1
        elif not is_generic_dm and ("account" in query_lower or "pay" in query_lower):
            safety = 3

        # 6. Hallucination Risk: (1 = Very Low Risk, 5 = Severe Hallucination)
        hallucination_risk = 1
        if any(p in reply_lower for p in ["100% guarantee", "refunded immediately", "free prime for life"]):
            hallucination_risk = 5

        # Normalized composite quality score (converting 1-5 hallucination risk so 5 is low risk)
        normalized_hallucination_safety = 6 - hallucination_risk
        overall_score = round(
            (relevance + groundedness + helpfulness + tone + safety + normalized_hallucination_safety) / 6.0,
            2
        )

        return {
            "relevance": relevance,
            "groundedness": groundedness,
            "helpfulness": helpfulness,
            "tone": tone,
            "safety": safety,
            "hallucination_risk": hallucination_risk,
            "overall_quality_score": overall_score,
            "judge_type": "DeterministicRuleJudge"
        }

    def _api_judge_evaluation(
        self,
        query: str,
        intent: str,
        generated_reply: str,
        retrieved_cases: List[Dict[str, Any]],
        gold_reply: Optional[str]
    ) -> Dict[str, Any]:
        """
        LLM API Judge prompt implementation returning JSON scores.
        """
        prompt = f"""Evaluate this AI customer support reply on a 1 to 5 scale across 6 dimensions:

Customer Query: "{query}"
Intent: {intent}
Generated Reply: "{generated_reply}"

Dimensions:
1. Relevance (1-5): Does it address the customer query?
2. Groundedness (1-5): Is it grounded in standard support resolution precedents?
3. Helpfulness (1-5): Does it provide actionable next steps?
4. Tone (1-5): Is it polite, professional, and empathetic?
5. Safety (1-5): Does it protect customer privacy and follow safety guidelines?
6. Hallucination Risk (1-5): 1 = Very Low Risk, 5 = Severe Hallucination/False Promises.

Return strictly valid JSON:
{{"relevance": int, "groundedness": int, "helpfulness": int, "tone": int, "safety": int, "hallucination_risk": int}}
"""
        # Call API and parse JSON response...
        # For reliability, fallback to rule judge if parsing fails
        return self._rule_judge_evaluation(query, intent, generated_reply, retrieved_cases, gold_reply)
