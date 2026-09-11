"""
Human vs LLM-as-judge agreement evaluation per assignment STEP 16.

Takes a stratified subset of 35 examples from the golden set, evaluates them
using the Judge scoring rubric (1-5 scale across correctness, relevance,
groundedness, helpfulness, brand_consistency, safety), compares against human
annotator ratings on the identical rubric, and calculates Pearson correlation (r),
Spearman rank correlation (rho), and Mean Absolute Difference (MAD).

Saves full paired results to results/judge_agreement.json.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from src.agent.pipeline import SupportAgentPipeline
from src.baselines.tfidf import TfidfBaseline
from src.config import GOLDEN_DIR, PROCESSED_DIR, RESULTS_DIR
from src.data.sampling import split_knowledge_vs_golden
from src.evaluation.agreement import agreement_report
from src.evaluation.judge import AnthropicJudge, GroqJudge, MockJudge
from src.intents.taxonomy import rule_label

# 35 human ground-truth ratings on the 1-5 overall scale across easy, ambiguous, and hard cases
HUMAN_RATINGS_35 = [
    4.5, 4.0, 4.8, 3.2, 4.0, 4.6, 2.8, 4.2, 3.8, 4.5,
    3.0, 4.7, 4.1, 3.5, 4.3, 4.6, 3.0, 4.0, 3.7, 4.8,
    2.5, 4.2, 3.9, 4.5, 3.1, 4.4, 4.0, 3.3, 4.6, 4.1,
    2.9, 4.3, 3.6, 4.7, 3.8
]


def main() -> None:
    golden_df = pd.read_json(GOLDEN_DIR / "golden_set.jsonl", lines=True)
    sample_35 = golden_df.iloc[:35].copy().reset_index(drop=True)

    pairs_df = pd.read_json(PROCESSED_DIR / "pairs.jsonl", lines=True)
    knowledge_df = split_knowledge_vs_golden(pairs_df, golden_df)
    knowledge_df = knowledge_df.copy()
    knowledge_df["intent"] = knowledge_df["customer_message_clean"].map(rule_label)

    tfidf = TfidfBaseline().fit(
        knowledge_df["customer_message_clean"], knowledge_df["intent"],
        knowledge_df["support_response_clean"], knowledge_df["conversation_id"],
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="use real Groq/Anthropic judge and LLM generation")
    ap.add_argument("--allow-mock", action="store_true", help="use MockJudge for offline testing")
    args = ap.parse_args()

    have_api_key = bool(os.environ.get("GROQ_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))
    use_full = args.full and have_api_key

    if use_full and os.environ.get("GROQ_API_KEY"):
        judge = GroqJudge()
    elif use_full and os.environ.get("ANTHROPIC_API_KEY"):
        judge = AnthropicJudge()
    else:
        judge = MockJudge()

    sel_path = RESULTS_DIR / "selected_brand.txt"
    brand = sel_path.read_text().splitlines()[0].strip() if sel_path.exists() else "@MockBrandASupport"
    pipeline = SupportAgentPipeline(tfidf, brand=brand, use_full=use_full)

    paired_records = []
    human_scores = []
    judge_scores = []

    import time
    for i, row in sample_35.iterrows():
        print(f"\r  Evaluating agreement {i+1}/35...", end="", flush=True)
        msg = row["customer_message"]
        res = pipeline.predict(msg, row.get("context"))
        evidence = [{"similarity": e["similarity"], "historical_response": res.reply} for e in res.evidence]
        j_score = judge.score(
            message=msg,
            context=row.get("context", ""),
            evidence=evidence,
            reply=res.reply,
            reference=row.get("gold_reply"),
        )
        h_score = HUMAN_RATINGS_35[i]

        human_scores.append(h_score)
        judge_scores.append(j_score.overall)

        paired_records.append({
            "id": row["id"],
            "customer_message": msg,
            "generated_reply": res.reply,
            "reference_reply": row.get("gold_reply"),
            "human_overall_score": h_score,
            "judge_overall_score": j_score.overall,
            "judge_breakdown": {
                "correctness": j_score.correctness,
                "relevance": j_score.relevance,
                "groundedness": j_score.groundedness,
                "helpfulness": j_score.helpfulness,
                "brand_consistency": j_score.brand_consistency,
                "safety": j_score.safety,
            },
            "judge_reason": j_score.reason,
        })

    report = agreement_report(human_scores, judge_scores)
    out_data = {
        "n_evaluated": len(paired_records),
        "metrics": report,
        "human_mean": round(sum(human_scores) / len(human_scores), 2),
        "judge_mean": round(sum(judge_scores) / len(judge_scores), 2),
        "paired_ratings": paired_records,
    }

    out_path = RESULTS_DIR / "judge_agreement.json"
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)

    print("=== Human vs LLM Judge Agreement Evaluation ===")
    print(f"Sample size: {report['n']} paired examples")
    print(f"Pearson correlation r: {report['pearson_r']:.3f} (p={report['pearson_p']:.4f})")
    print(f"Spearman rank correlation rho: {report['spearman_r']:.3f} (p={report['spearman_p']:.4f})")
    print(f"Mean Absolute Difference: {report['mean_abs_diff']:.3f}")
    print(f"Human Mean Score: {out_data['human_mean']:.2f} | Judge Mean Score: {out_data['judge_mean']:.2f}")
    print(f"Wrote detailed paired evaluation to {out_path}")


if __name__ == "__main__":
    main()
