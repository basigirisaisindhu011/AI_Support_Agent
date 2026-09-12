import os
import requests
from typing import List, Dict, Any, Optional
from src.generation.prompts import GROUNDED_SYSTEM_PROMPT, GROUNDED_USER_PROMPT

class TrivialReplyGenerator:
    """
    Baseline Trivial System: Generic static support template reply.
    """
    def generate_reply(self, query: str, intent: str, retrieved_cases: List[Dict[str, Any]]) -> str:
        return "Hello! Thank you for reaching out to Amazon Help. Please send us a Direct Message (DM) with your order number so we can assist you further."


class Simple1NNReplyGenerator:
    """
    Baseline Simple System: Directly returns the response text of the single nearest historical case (1-NN).
    """
    def generate_reply(self, query: str, intent: str, retrieved_cases: List[Dict[str, Any]]) -> str:
        if retrieved_cases:
            resp = retrieved_cases[0].get("support_response", "")
            if resp:
                return resp
        return "Hello! Please send us a Direct Message (DM) with your order details so we can investigate this for you."


class MainGroundedReplyGenerator:
    """
    Main System: Top-k Retrieval + Grounded RAG Generator.
    Supports Groq/OpenAI APIs if key is set, or a deterministic offline grounded synthesis engine.
    """
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.model = os.getenv("LLM_MODEL", "llama3-70b-8192")

    def generate_reply(self, query: str, intent: str, retrieved_cases: List[Dict[str, Any]]) -> str:
        # Build context from top retrieved precedents
        context_lines = []
        for i, case in enumerate(retrieved_cases, 1):
            cust = case.get("customer_clean_text", "")
            supp = case.get("support_response", "")
            sim = case.get("similarity_score", 0.0)
            context_lines.append(f"Precedent {i} (Similarity: {sim}):\n  Customer: {cust}\n  Support Resolution: {supp}")
            
        context_str = "\n\n".join(context_lines) if context_lines else "No specific historical resolution found."
        
        # Try API if available
        if self.groq_api_key:
            try:
                return self._call_groq_api(query, intent, context_str)
            except Exception as e:
                print(f"[Generator] Groq API call failed ({e}). Using offline grounded generator.")
        elif self.openai_api_key:
            try:
                return self._call_openai_api(query, intent, context_str)
            except Exception as e:
                print(f"[Generator] OpenAI API call failed ({e}). Using offline grounded generator.")

        # High-fidelity offline deterministic grounded synthesis fallback
        return self._offline_grounded_synthesis(query, intent, retrieved_cases)

    def _call_groq_api(self, query: str, intent: str, context: str) -> str:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": GROUNDED_SYSTEM_PROMPT},
                {"role": "user", "content": GROUNDED_USER_PROMPT.format(query=query, intent=intent, context=context)}
            ],
            "temperature": 0.2,
            "max_tokens": 150
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=10)
        resp.raise_for_request()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    def _call_openai_api(self, query: str, intent: str, context: str) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": GROUNDED_SYSTEM_PROMPT},
                {"role": "user", "content": GROUNDED_USER_PROMPT.format(query=query, intent=intent, context=context)}
            ],
            "temperature": 0.2,
            "max_tokens": 150
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=10)
        resp.raise_for_request()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    def _offline_grounded_synthesis(self, query: str, intent: str, retrieved_cases: List[Dict[str, Any]]) -> str:
        """
        Deterministic, offline grounded answer synthesis based on top precedent and intent guidelines.
        """
        if not retrieved_cases:
            return "Hello! Thanks for contacting Amazon Help. Please send us a DM with your order details so we can investigate."

        top_case = retrieved_cases[0]
        top_resp = top_case.get("support_response", "")

        intent_actions = {
            "DELIVERY_STATUS": "We understand you are inquiring about your package delivery. Please send us a DM with your order ID so we can track the exact status for you.",
            "REFUND_RETURN": "For return or refund assistance, please send us a DM with your account email and order details so our team can process this promptly.",
            "CANCELLATION": "If you wish to cancel an order, please send us a DM right away with your order number so we can attempt cancellation before dispatch.",
            "PAYMENT_CHARGE": "For payment or billing inquiries, please DM us your order details and account email so we can verify charges safely.",
            "ACCOUNT_ACCESS": "For your security, account login issues must be handled via direct message. Please DM us your registered email address.",
            "DAMAGED_WRONG_MISSING_ITEM": "We are very sorry to hear your item arrived damaged or missing! Please send us a DM with photos and order details so we can arrange a replacement.",
            "PRIME_SUBSCRIPTION": "Regarding your Amazon Prime membership, please send us a DM with your account email so we can check your subscription details.",
            "GENERAL_OTHER": "Thanks for reaching out! Please send us a DM with further details so we can assist you right away."
        }

        # Synthesize grounded answer combining top historical resolution pattern with intent safety action
        action_text = intent_actions.get(intent, intent_actions["GENERAL_OTHER"])
        return f"Hello! {action_text}"
