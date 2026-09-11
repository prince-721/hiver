from src.predict import build_pipeline


def test_pipeline_end_to_end_light_mode():
    pipeline = build_pipeline(use_full=False)
    result = pipeline.predict("My order says delivered but I never received it, order #99887")
    assert result.mode == "light"
    assert result.intent  # non-empty
    assert result.escalation_decision in {"AUTO_HANDLE", "ESCALATE"}
    assert isinstance(result.evidence, list)
    assert result.reply  # non-empty


def test_pipeline_escalates_high_risk():
    pipeline = build_pipeline(use_full=False)
    result = pipeline.predict("someone hacked my account and stole my card details")
    assert result.escalation_decision == "ESCALATE"
