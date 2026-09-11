from src.evaluation.agreement import agreement_report


def test_agreement_perfect_correlation():
    human = [1, 2, 3, 4, 5]
    judge = [1, 2, 3, 4, 5]
    r = agreement_report(human, judge)
    assert r["pearson_r"] == 1.0
    assert r["spearman_r"] == 1.0
    assert r["mean_abs_diff"] == 0.0


def test_agreement_computes_on_noisy_data():
    human = [2, 3, 3, 4, 5, 2, 4]
    judge = [2, 3, 4, 4, 4, 3, 5]
    r = agreement_report(human, judge)
    assert 0 <= r["pearson_r"] <= 1
    assert r["n"] == 7
