from src.intents.classifier import IntentClassifier


def test_intent_classifier_fit_and_predict():
    classifier = IntentClassifier()
    messages = [
        "cannot log into my account",
        "password reset link expired",
        "need a refund for broken item",
        "want my money back please",
    ]
    labels = ["login_problem", "login_problem", "refund_request", "refund_request"]
    classifier.fit(messages, labels)

    pred = classifier.predict("locked out of account")
    assert "intent" in pred
    assert pred["intent"] in ["login_problem", "refund_request"]
    assert 0.0 <= pred["confidence"] <= 1.0
    assert isinstance(pred["top_candidates"], list)
    assert len(pred["top_candidates"]) <= 3
