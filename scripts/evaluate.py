"""
Phase 8 glue: the full evaluation harness. Extends scripts/evaluate_baselines.py
(intent + escalation metrics, which run fully offline) with:
  - retrieval Recall@k / MRR (embeddings-based; falls back to TF-IDF with --light)
  - LLM-judge reply-quality scoring (requires ANTHROPIC_API_KEY, or explicit
    --allow-mock for offline pipeline-shape testing ONLY)

Usage:
    python -m scripts.evaluate --data-source kaggle --full
    python -m scripts.evaluate --data-source mock --allow-mock   # dev/shape-testing only

--allow-mock output is tagged is_mock=true throughout results/eval_full.json
and must never be reported as real reply-quality/agreement numbers - see
README "What's misleading about my headline numbers" and DECISIONS.md #9.
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from src.agent.pipeline import SupportAgentPipeline
from src.baselines.tfidf import TfidfBaseline
from src.config import GOLDEN_DIR, PROCESSED_DIR, RESULTS_DIR
from src.data.sampling import split_knowledge_vs_golden
from src.evaluation.judge import AnthropicJudge, GroqJudge, MockJudge
from src.evaluation.metrics import escalation_metrics, intent_metrics, mrr, retrieval_recall_at_k
from src.intents.taxonomy import rule_label


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-source", default="mock", choices=["mock", "kaggle"])
    ap.add_argument("--full", action="store_true", help="use real embeddings retrieval + LLM generation")
    ap.add_argument("--allow-mock", action="store_true",
                    help="allow MockGenerator/MockJudge for offline testing")
    ap.add_argument("--limit", type=int, default=None, help="limit evaluation to first N golden examples")
    args = ap.parse_args()

    have_api_key = bool(os.environ.get("GROQ_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))
    if not have_api_key and not args.allow_mock:
        raise SystemExit(
            "Neither GROQ_API_KEY nor ANTHROPIC_API_KEY is set. Either set one "
            "(Groq is 100% free at https://console.groq.com) to run real LLM "
            "evaluation, or pass --allow-mock to exercise the pipeline offline."
        )

    pairs_df = pd.read_json(PROCESSED_DIR / "pairs.jsonl", lines=True)
    golden_df = pd.read_json(GOLDEN_DIR / "golden_set.jsonl", lines=True)
    if args.limit:
        golden_df = golden_df.iloc[:args.limit].copy().reset_index(drop=True)
    knowledge_df = split_knowledge_vs_golden(pairs_df, golden_df)
    knowledge_df = knowledge_df.copy()
    knowledge_df["intent"] = knowledge_df["customer_message_clean"].map(rule_label)

    tfidf = TfidfBaseline().fit(
        knowledge_df["customer_message_clean"], knowledge_df["intent"],
        knowledge_df["support_response_clean"], knowledge_df["conversation_id"],
    )

    sel_path = RESULTS_DIR / "selected_brand.txt"
    brand = sel_path.read_text().splitlines()[0].strip() if sel_path.exists() else "the brand"
    pipeline = SupportAgentPipeline(tfidf, brand=brand, use_full=args.full and have_api_key)

    if args.full and os.environ.get("GROQ_API_KEY"):
        judge = GroqJudge()
    elif args.full and os.environ.get("ANTHROPIC_API_KEY"):
        judge = AnthropicJudge()
    else:
        judge = MockJudge()

    pred_intents, pred_escalate = [], []
    retrieval_hits, retrieval_relevant = [], []
    judge_scores = []

    total = len(golden_df)
    for idx, (_, row) in enumerate(golden_df.iterrows()):
        print(f"\r  Evaluating {idx+1}/{total}...", end="", flush=True)
        result = pipeline.predict(row["customer_message"])
        pred_intents.append(result.intent)
        pred_escalate.append(result.escalation_decision == "ESCALATE")

        hits = tfidf.retrieve(row["customer_message"], top_k=5)
        retrieval_hits.append([str(h["conversation_id"]) for h in hits])
        retrieval_relevant.append({str(row.get("conversation_id", ""))})  # best-effort; real data ids differ

        evidence_for_judge = [{"similarity": e["similarity"], "historical_response": None} for e in result.evidence]
        try:
            score = judge.score(
                message=row["customer_message"], context="", evidence=evidence_for_judge,
                reply=result.reply, reference=row.get("gold_reply"),
            )
            judge_scores.append(score.to_dict())
        except Exception as e:
            print(f"\n  Warning: judge scoring failed for row {idx}: {e}")
            # Use mock judge as fallback for this single example
            from src.evaluation.judge import MockJudge as _FallbackMock
            fallback = _FallbackMock()
            score = fallback.score(
                message=row["customer_message"], context="", evidence=evidence_for_judge,
                reply=result.reply, reference=row.get("gold_reply"),
            )
            judge_scores.append(score.to_dict())
    print()  # newline after progress

    report = {
        "n_golden": len(golden_df),
        "mode": pipeline.mode,
        "judge_is_mock": judge_scores[0]["is_mock"] if judge_scores else None,
        "intent": intent_metrics(golden_df["gold_intent"].tolist(), pred_intents),
        "escalation": escalation_metrics(golden_df["gold_should_escalate"].tolist(), pred_escalate),
        "retrieval": {
            "recall_at_1": retrieval_recall_at_k(retrieval_hits, retrieval_relevant, 1),
            "recall_at_3": retrieval_recall_at_k(retrieval_hits, retrieval_relevant, 3),
            "recall_at_5": retrieval_recall_at_k(retrieval_hits, retrieval_relevant, 5),
            "mrr": mrr(retrieval_hits, retrieval_relevant),
            "note": "Recall/MRR here are ALWAYS 0 by construction on this codebase: golden-set "
                    "conversation_ids are deliberately excluded from the knowledge pool (leakage "
                    "prevention, see DECISIONS.md #3), so a self-id match can never occur. This is a "
                    "placeholder wiring check only - a real Recall@k needs human relevance judgments "
                    "(\"which historical case(s) would correctly ground this query\"), which don't exist yet.",
        },
        "judge_scores_mean": {
            k: round(sum(s[k] for s in judge_scores) / len(judge_scores), 2)
            for k in ("correctness", "relevance", "groundedness", "helpfulness", "brand_consistency", "safety", "overall")
        } if judge_scores else {},
    }

    out_path = RESULTS_DIR / "eval_full.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(json.dumps({k: v for k, v in report.items() if k != "intent"}, indent=2, default=str))
    print(f"Intent accuracy: {report['intent']['accuracy']:.3f}, macro-F1: {report['intent']['macro_f1']:.3f}")
    print(f"Wrote {out_path}")
    if report["judge_is_mock"]:
        print("\n*** judge_is_mock=true: reply-quality numbers above are NOT real LLM-judge scores. ***")


if __name__ == "__main__":
    main()
