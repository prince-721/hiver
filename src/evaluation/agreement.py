"""
Computes agreement between human ratings and LLM-judge ratings on the
same rubric. This module is fully executable here (scipy has no network
dependency) - what's NOT executable in this sandbox is producing the
actual human and LLM-judge ratings themselves (the latter needs API
access; the former needs an actual human, i.e. you).
"""
from __future__ import annotations

from scipy.stats import pearsonr, spearmanr


def agreement_report(human_scores: list[float], judge_scores: list[float]) -> dict:
    if len(human_scores) != len(judge_scores):
        raise ValueError("human_scores and judge_scores must be the same length")
    if len(human_scores) < 3:
        raise ValueError("Need at least 3 paired ratings to compute correlation meaningfully")

    pearson_r, pearson_p = pearsonr(human_scores, judge_scores)
    spearman_r, spearman_p = spearmanr(human_scores, judge_scores)
    mean_abs_diff = sum(abs(h - j) for h, j in zip(human_scores, judge_scores)) / len(human_scores)

    return {
        "n": len(human_scores),
        "pearson_r": round(float(pearson_r), 3),
        "pearson_p": round(float(pearson_p), 4),
        "spearman_r": round(float(spearman_r), 3),
        "spearman_p": round(float(spearman_p), 4),
        "mean_abs_diff": round(mean_abs_diff, 3),
    }
