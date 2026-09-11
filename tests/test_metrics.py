from src.evaluation.metrics import escalation_metrics, intent_metrics, mrr, retrieval_recall_at_k


def test_intent_metrics_perfect():
    y = ["a", "b", "a", "c"]
    m = intent_metrics(y, y)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0


def test_escalation_metrics_false_auto_handle_counted():
    y_true = [True, False, True, False]
    y_pred = [False, False, True, True]  # index 0: gold escalate, predicted auto-handle -> dangerous
    m = escalation_metrics(y_true, y_pred)
    assert m["false_auto_handle_rate"] == 0.25
    assert m["n"] == 4


def test_retrieval_recall_and_mrr():
    hits = [["docA", "docB", "docC"], ["docX", "docY"]]
    relevant = [{"docB"}, {"docZ"}]
    assert retrieval_recall_at_k(hits, relevant, k=3) == 0.5
    assert retrieval_recall_at_k(hits, relevant, k=1) == 0.0
    m = mrr(hits, relevant)
    assert round(m, 3) == round((1 / 2) / 2, 3)  # docB at rank 2 for query 1, 0 for query 2
