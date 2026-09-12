from src.intents.baseline import MajorityClassBaseline, TfIdfBaseline
from src.intents.classifier import IntentClassifier
from src.intents.train import evaluate_classifier_model

def test_majority_baseline():
    X_train = ["order status", "delayed delivery", "cancel order"]
    y_train = ["DELIVERY_STATUS", "DELIVERY_STATUS", "CANCELLATION"]
    
    b0 = MajorityClassBaseline()
    b0.fit(X_train, y_train)
    preds = b0.predict(["random text", "another query"])
    assert preds == ["DELIVERY_STATUS", "DELIVERY_STATUS"]

def test_tfidf_baseline():
    X_train = ["where is my delivery package", "cancel my order immediately", "want a refund for return"]
    y_train = ["DELIVERY_STATUS", "CANCELLATION", "REFUND_RETURN"]
    
    b1 = TfIdfBaseline()
    b1.fit(X_train, y_train)
    pred = b1.predict(["package delivery status"])
    assert pred[0] in ["DELIVERY_STATUS", "CANCELLATION", "REFUND_RETURN"]

def test_intent_classifier():
    X_train = ["where is my delivery package", "cancel my order immediately", "want a refund for return"]
    y_train = ["DELIVERY_STATUS", "CANCELLATION", "REFUND_RETURN"]
    
    clf = IntentClassifier()
    clf.fit(X_train, y_train)
    intent, conf, _ = clf.predict_with_confidence("cancel my subscription order")
    assert intent in y_train
    assert 0.0 <= conf <= 1.0
