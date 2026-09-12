from typing import Dict, Any, Optional
from src.config import get_config
from src.intents.classifier import IntentClassifier
from src.retrieval.index import VectorSearchIndex
from src.retrieval.retriever import HistoricalRetriever
from src.routing.escalation import EscalationRouter
from src.generation.generator import MainGroundedReplyGenerator
from src.evaluation.judge import LLMSupportJudge

class SupportAgentPipeline:
    """
    End-to-End Pipeline for Hiver AI Support Agent.
    Executes Intent Classification -> Historical Retrieval -> Escalation Routing -> Grounded Generation -> Judge Scoring.
    """
    def __init__(
        self,
        classifier: Optional[IntentClassifier] = None,
        retriever: Optional[HistoricalRetriever] = None,
        router: Optional[EscalationRouter] = None,
        generator: Optional[MainGroundedReplyGenerator] = None,
        judge: Optional[LLMSupportJudge] = None
    ):
        self.config = get_config()
        self.classifier = classifier or IntentClassifier()
        
        # Load vector index if available
        if retriever is None:
            if self.config.vector_index_path.exists():
                v_index = VectorSearchIndex.load(self.config.vector_index_path)
                self.retriever = HistoricalRetriever(index=v_index)
            else:
                self.retriever = HistoricalRetriever(index=None)
        else:
            self.retriever = retriever
            
        self.router = router or EscalationRouter(config=self.config)
        self.generator = generator or MainGroundedReplyGenerator()
        self.judge = judge or LLMSupportJudge()

    def process_query(self, query: str, query_tweet_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Processes an incoming customer query through the full agent pipeline.
        """
        # Step 1: Intent Classification & Confidence Score
        if self.classifier.is_fitted:
            predicted_intent, confidence, all_probs = self.classifier.predict_with_confidence(query)
        else:
            from src.intents.taxonomy import suggest_intent_from_keywords
            predicted_intent = suggest_intent_from_keywords(query)
            confidence = 0.75
            all_probs = {predicted_intent: 0.75}

        # Step 2: Historical Resolution Retrieval with Leak Masking
        retrieved_cases = self.retriever.retrieve(
            query=query,
            top_k=self.config.top_k,
            query_tweet_id=query_tweet_id
        )

        # Step 3: Auto-Handle vs Escalation Routing Decision
        routing = self.router.evaluate_routing(
            query=query,
            predicted_intent=predicted_intent,
            confidence=confidence,
            retrieved_cases=retrieved_cases
        )

        # Step 4: Grounded Reply Generation
        generated_reply = self.generator.generate_reply(
            query=query,
            intent=predicted_intent,
            retrieved_cases=retrieved_cases
        )

        # Step 5: Judge Evaluation
        judge_score = self.judge.evaluate_reply(
            query=query,
            intent=predicted_intent,
            generated_reply=generated_reply,
            retrieved_cases=retrieved_cases
        )

        return {
            "query": query,
            "predicted_intent": predicted_intent,
            "confidence": confidence,
            "all_intent_probabilities": all_probs,
            "routing": routing,
            "action": routing["action"],
            "retrieved_cases": retrieved_cases,
            "generated_reply": generated_reply,
            "judge_score": judge_score
        }
