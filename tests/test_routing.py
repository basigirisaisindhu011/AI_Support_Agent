from src.routing.escalation import EscalationRouter

def test_angry_customer_delivery_auto_handled():
    """
    Angry customer with straightforward delivery query should NOT be automatically escalated!
    """
    router = EscalationRouter()
    query = "I am extremely angry and furious that my delivery package is late!"
    res = router.evaluate_routing(
        query=query,
        predicted_intent="DELIVERY_STATUS",
        confidence=0.92,
        retrieved_cases=[{"similarity_score": 0.85}]
    )
    assert res["action"] == "AUTO_HANDLE"

def test_legal_threat_escalated():
    """
    Query mentioning legal action / lawyer MUST escalate immediately.
    """
    router = EscalationRouter()
    query = "If I don't get my refund today I will hire a lawyer to sue Amazon!"
    res = router.evaluate_routing(
        query=query,
        predicted_intent="REFUND_RETURN",
        confidence=0.88,
        retrieved_cases=[{"similarity_score": 0.70}]
    )
    assert res["action"] == "ESCALATE_HUMAN"
    assert res["should_escalate"] is True
