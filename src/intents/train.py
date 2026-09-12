import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from src.intents.baseline import MajorityClassBaseline, TfIdfBaseline
from src.intents.classifier import IntentClassifier

def evaluate_classifier_model(model: Any, X_test: List[str], y_test: List[str]) -> Dict[str, float]:
    """
    Evaluates classifier predictions against ground truth y_test.
    Headline metric: Macro F1.
    """
    y_pred = model.predict(X_test)
    
    acc = float(accuracy_score(y_test, y_pred))
    macro_prec = float(precision_score(y_test, y_pred, average='macro', zero_division=0))
    macro_rec = float(recall_score(y_test, y_pred, average='macro', zero_division=0))
    macro_f1 = float(f1_score(y_test, y_pred, average='macro', zero_division=0))
    weighted_f1 = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
    
    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "macro_f1": round(macro_f1, 4),  # Headline classification metric!
        "weighted_f1": round(weighted_f1, 4)
    }

def train_and_compare_classifiers(
    X_train: List[str],
    y_train: List[str],
    X_test: List[str],
    y_test: List[str]
) -> Tuple[Dict[str, Dict[str, float]], IntentClassifier]:
    """
    Trains Baseline 0, Baseline 1, and Main SentenceTransformer Classifier.
    Returns comparison metrics dictionary and fitted main classifier model.
    """
    results = {}
    
    # 1. Baseline 0: Majority Class
    b0 = MajorityClassBaseline()
    b0.fit(X_train, y_train)
    results["baseline_0_majority"] = evaluate_classifier_model(b0, X_test, y_test)
    
    # 2. Baseline 1: TF-IDF + Logistic Regression
    b1 = TfIdfBaseline()
    b1.fit(X_train, y_train)
    results["baseline_1_tfidf"] = evaluate_classifier_model(b1, X_test, y_test)
    
    # 3. Main Model: SentenceTransformer + Logistic Regression
    main_clf = IntentClassifier()
    main_clf.fit(X_train, y_train)
    results["main_model_sentence_transformer"] = evaluate_classifier_model(main_clf, X_test, y_test)
    
    return results, main_clf
