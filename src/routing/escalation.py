import re
from typing import Dict, Any, List, Optional
from src.config import get_config

class EscalationRouter:
    """
    Auto-Handle vs Human Escalation Router.
    Enforces risk-aware routing based on intent confidence, retrieval evidence,
    security/legal risks, and policy boundaries.
    
    NOTE: Negative sentiment is NOT an automatic escalation trigger on its own.
    """
    def __init__(self, config: Optional[Any] = None):
        self.config = config or get_config()
        self.min_confidence = self.config.min_confidence_threshold
        self.min_retrieval_sim = self.config.min_retrieval_sim_threshold
        self.sensitive_intents = set(self.config.sensitive_intents)
        self.security_legal_keywords = [kw.lower() for kw in self.config.security_legal_keywords]

    def evaluate_routing(
        self,
        query: str,
        predicted_intent: str,
        confidence: float,
        retrieved_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Determines whether to AUTO_HANDLE or ESCALATE_HUMAN.
        Returns dictionary with action, risk_score, and primary_reason.
        """
        reasons = []
        risk_score = 0.0
        query_lower = query.lower()

        # Signal 1: Security & Legal Keywords (High Risk)
        found_legal = [kw for kw in self.security_legal_keywords if kw in query_lower]
        if found_legal:
            reasons.append(f"Contains security/legal risk keywords: {found_legal}")
            risk_score += 0.8

        # Signal 2: Sensitive Financial / Account Access Intents
        if predicted_intent in self.sensitive_intents:
            reasons.append(f"Intent '{predicted_intent}' involves sensitive financial or credential actions")
            risk_score += 0.5

        # Signal 3: Low Intent Classification Confidence
        if confidence < self.min_confidence:
            reasons.append(f"Low intent classification confidence ({confidence:.2f} < {self.min_confidence})")
            risk_score += 0.4

        # Signal 4: Weak Retrieval Evidence
        top_sim = retrieved_cases[0].get("similarity_score", 0.0) if retrieved_cases else 0.0
        if top_sim < self.min_retrieval_sim:
            reasons.append(f"Weak historical retrieval evidence (similarity {top_sim:.2f} < {self.min_retrieval_sim})")
            risk_score += 0.4

        # Signal 5: General/Other fallback intent with moderate confidence
        if predicted_intent == "GENERAL_OTHER" and confidence < 0.8:
            reasons.append("Unclear/General query intent without high confidence pattern match")
            risk_score += 0.3

        # Decision Threshold Logic
        should_escalate = (risk_score >= 0.5) or (len(reasons) >= 2) or bool(found_legal)

        action = "ESCALATE_HUMAN" if should_escalate else "AUTO_HANDLE"
        primary_reason = "; ".join(reasons) if reasons else "High intent confidence and strong historical retrieval evidence."

        return {
            "action": action,
            "should_escalate": should_escalate,
            "risk_score": round(min(1.0, risk_score), 4),
            "reasons": reasons,
            "primary_reason": primary_reason,
            "confidence": round(confidence, 4),
            "top_retrieval_sim": round(top_sim, 4)
        }
