from typing import Dict, List

INTENT_TAXONOMY: List[str] = [
    "DELIVERY_STATUS",
    "REFUND_RETURN",
    "CANCELLATION",
    "PAYMENT_CHARGE",
    "ACCOUNT_ACCESS",
    "DAMAGED_WRONG_MISSING_ITEM",
    "PRIME_SUBSCRIPTION",
    "GENERAL_OTHER"
]

INTENT_DESCRIPTIONS: Dict[str, str] = {
    "DELIVERY_STATUS": "Queries regarding package tracking, shipment delays, delivery dates, or missing tracking info.",
    "REFUND_RETURN": "Requests or status checks for product returns, refund timelines, or return labels.",
    "CANCELLATION": "Requests to cancel an active order before delivery or order cancellation confirmations.",
    "PAYMENT_CHARGE": "Issues with billing, double charges, failed transactions, gift cards, or payment methods.",
    "ACCOUNT_ACCESS": "Difficulty logging in, password resets, account locking, OTP issues, or unauthorized access.",
    "DAMAGED_WRONG_MISSING_ITEM": "Reports of damaged goods, defective products, wrong item delivered, or missing items in box.",
    "PRIME_SUBSCRIPTION": "Questions or cancellation requests regarding Amazon Prime membership, video streaming, or Prime benefits.",
    "GENERAL_OTHER": "General questions, customer feedback, praise, store inquiries, or non-actionable complaints."
}

# Rule-based candidate heuristic keywords for initial dataset labelling assistance
KEYWORD_RULES: Dict[str, List[str]] = {
    "DELIVERY_STATUS": ["where is my order", "track", "delivery", "delivered", "shipping", "shipped", "carrier", "late", "eta", "driver", "package"],
    "REFUND_RETURN": ["refund", "return", "return label", "money back", "reimburse", "credited"],
    "CANCELLATION": ["cancel", "cancellation", "cancelling", "stop order"],
    "PAYMENT_CHARGE": ["charged", "payment", "billing", "card", "double charge", "invoice", "overcharge", "gift card"],
    "ACCOUNT_ACCESS": ["login", "password", "account", "otp", "sign in", "locked", "unauthorized", "hacked", "email reset"],
    "DAMAGED_WRONG_MISSING_ITEM": ["damaged", "broken", "wrong item", "missing item", "defective", "torn", "empty box", "faulty"],
    "PRIME_SUBSCRIPTION": ["prime", "membership", "subscription", "prime video", "auto renew"]
}

def suggest_intent_from_keywords(text: str) -> str:
    """
    Suggests a candidate intent based on domain keyword rules to assist manual golden set labelling.
    """
    text_lower = text.lower()
    for intent, keywords in KEYWORD_RULES.items():
        if any(kw in text_lower for kw in keywords):
            return intent
    return "GENERAL_OTHER"
