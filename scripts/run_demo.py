import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import SupportAgentPipeline

def main():
    print("==========================================================")
    print("      HIVER AI SUPPORT AGENT - INTERACTIVE CLI DEMO       ")
    print("==========================================================")
    print("Type your customer support query below or type 'exit' to quit.\n")

    pipeline = SupportAgentPipeline()

    sample_queries = [
        "Where is my package? Tracking says delivered to reception but I have no reception desk.",
        "I was double charged $49.99 on my Visa credit card for order 402-9182312.",
        "Someone hacked my account and changed the email address!"
    ]

    print("--- Running Sample Queries ---")
    for q in sample_queries:
        print(f"\n[Customer Query]: {q}")
        res = pipeline.process_query(q)
        print(f"  * Detected Intent: {res['predicted_intent']} (Confidence: {res['confidence']:.2f})")
        print(f"  * Routing Action:  {res['action']} (Risk Score: {res['routing']['risk_score']:.2f})")
        print(f"  * Primary Reason:  {res['routing']['primary_reason']}")
        print(f"  * Generated Reply: {res['generated_reply']}")
        print(f"  * Judge Score:     {res['judge_score']['overall_quality_score']}/5.0")
    print("\n----------------------------------------------------------\n")

    while True:
        try:
            user_input = input("Enter Customer Tweet Query > ").strip()
            if not user_input or user_input.lower() in ["exit", "quit"]:
                print("Exiting demo. Goodbye!")
                break

            res = pipeline.process_query(user_input)
            print("\n" + "="*50)
            print(f"Detected Intent: {res['predicted_intent']} (Confidence: {res['confidence']:.2f})")
            print(f"Routing Decision: {res['action']} (Risk Score: {res['routing']['risk_score']:.2f})")
            print(f"Reason: {res['routing']['primary_reason']}")
            print(f"\n[Generated Grounded Reply]:\n{res['generated_reply']}")
            print(f"\n[LLM Judge Quality Score]: {res['judge_score']['overall_quality_score']}/5.0")
            print("="*50 + "\n")
        except KeyboardInterrupt:
            print("\nExiting demo.")
            break

if __name__ == "__main__":
    main()
