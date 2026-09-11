from src.escalation.policy import decide


def test_high_risk_always_escalates_regardless_of_confidence():
    d = decide("this account was hacked, someone stole my money", "technical_problem", 0.99, 0.99, 5)
    assert d.decision == "ESCALATE"
    assert "risk" in d.reason.lower()


def test_explicit_human_request_escalates():
    d = decide("I want to talk to a human please", "complaint", 0.9, 0.9, 5)
    assert d.decision == "ESCALATE"


def test_no_evidence_escalates():
    d = decide("some obscure issue", "other", 0.9, None, 0)
    assert d.decision == "ESCALATE"


def test_confident_well_grounded_non_billing_auto_handles():
    d = decide("app crashes on the payments tab", "technical_problem", 0.9, 0.9, 3)
    assert d.decision == "AUTO_HANDLE"


def test_billing_always_escalates_even_with_high_confidence():
    d = decide("charged twice for order 123", "billing_issue", 0.95, 0.95, 3)
    assert d.decision == "ESCALATE"
