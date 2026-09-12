import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config
from src.evaluation.run_eval import run_master_evaluation

def main():
    """
    One-click reproduction script for hiring managers & technical reviewers.
    Loads the fixed 200-example manually annotated Golden Set and executes
    master pipeline benchmark evaluation.
    """
    config = get_config()
    print("[Reproduce] Starting automated benchmark reproduction on fixed Golden Set...")
    
    # Ensure golden set exists
    if not config.golden_set_path.exists():
        print(f"[Reproduce] Golden set file missing at {config.golden_set_path}. Generating fixed 200-example set...")
        from scripts.build_golden_set import main as build_golden_main
        build_golden_main()
        
    master_results = run_master_evaluation()
    print("\n[Reproduce] Results successfully reproduced and saved to:")
    print(f"  - {config.results_json_path}")
    print(f"  - {config.predictions_csv_path}")
    print(f"  - {config.failure_csv_path}")

if __name__ == "__main__":
    main()
