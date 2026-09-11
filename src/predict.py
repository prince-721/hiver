"""
CLI: python -m src.predict --message "..." [--context "..."] [--full]

Fits the TF-IDF baseline + rule-labeled intents on the current knowledge
pool (excluding golden set) each run for simplicity/reproducibility -
for repeated interactive use, cache this via scripts/build_index.py in a
future iteration (see DECISIONS.md).
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from src.agent.pipeline import SupportAgentPipeline
from src.baselines.tfidf import TfidfBaseline
from src.config import GOLDEN_DIR, PROCESSED_DIR, RESULTS_DIR
from src.data.sampling import split_knowledge_vs_golden
from src.intents.taxonomy import rule_label


def build_pipeline(use_full: bool = False) -> SupportAgentPipeline:
    pairs_df = pd.read_json(PROCESSED_DIR / "pairs.jsonl", lines=True)
    golden_path = GOLDEN_DIR / "golden_set.jsonl"
    if golden_path.exists():
        golden_df = pd.read_json(golden_path, lines=True)
        knowledge_df = split_knowledge_vs_golden(pairs_df, golden_df)
    else:
        knowledge_df = pairs_df

    knowledge_df = knowledge_df.copy()
    knowledge_df["intent"] = knowledge_df["customer_message_clean"].map(rule_label)

    tfidf = TfidfBaseline().fit(
        knowledge_df["customer_message_clean"], knowledge_df["intent"],
        knowledge_df["support_response_clean"], knowledge_df["conversation_id"],
    )

    sel_path = RESULTS_DIR / "selected_brand.txt"
    brand = sel_path.read_text().splitlines()[0].strip() if sel_path.exists() else "the brand"
    return SupportAgentPipeline(tfidf, brand=brand, use_full=use_full)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--message", required=True)
    ap.add_argument("--context", default="")
    ap.add_argument("--full", action="store_true", help="use real embeddings+LLM if available")
    args = ap.parse_args()

    pipeline = build_pipeline(use_full=args.full)
    result = pipeline.predict(args.message, args.context)

    print(json.dumps({
        "intent": result.intent,
        "intent_confidence": result.intent_confidence,
        "reply": result.reply,
        "escalation": {
            "decision": result.escalation_decision,
            "reason": result.escalation_reason,
            "confidence": result.escalation_confidence,
        },
        "evidence": result.evidence,
        "mode": result.mode,
    }, indent=2))


if __name__ == "__main__":
    main()
