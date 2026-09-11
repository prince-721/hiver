"""
Phase 5+8 (partial): evaluate the two non-LLM baselines against the golden
set. This is the part of the evaluation harness that needs no API key and
no embedding model - it runs fully offline and produces REAL numbers.

The full evaluate.py (added later) extends this with the LLM-based system
and LLM-judge reply-quality scores, which require ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import json

import pandas as pd

from src.baselines.majority import MajorityBaseline
from src.baselines.tfidf import TfidfBaseline
from src.config import PROCESSED_DIR, GOLDEN_DIR, RESULTS_DIR
from src.data.sampling import split_knowledge_vs_golden
from src.escalation.policy import decide
from src.evaluation.metrics import escalation_metrics, intent_metrics
from src.intents.taxonomy import rule_label


def main() -> None:
    pairs_df = pd.read_json(PROCESSED_DIR / "pairs.jsonl", lines=True)
    golden_df = pd.read_json(GOLDEN_DIR / "golden_set.jsonl", lines=True)

    knowledge_df = split_knowledge_vs_golden(pairs_df, golden_df)
    print(f"Knowledge/train pool: {len(knowledge_df)} pairs "
          f"({knowledge_df.attrs.get('n_removed_for_golden', 0)} removed as held-out golden examples).")

    knowledge_df = knowledge_df.copy()
    knowledge_df["intent"] = knowledge_df["customer_message_clean"].map(rule_label)

    results = {}

    # ---- Baseline 1: majority ----
    maj = MajorityBaseline().fit(knowledge_df["intent"])
    maj_pred_intents = [maj.predict_intent(m)["intent"] for m in golden_df["customer_message"]]
    maj_pred_escalate = [maj.predict_escalation()["decision"] == "ESCALATE" for _ in golden_df["customer_message"]]

    results["majority_baseline"] = {
        "intent": intent_metrics(golden_df["gold_intent"].tolist(), maj_pred_intents),
        "escalation": escalation_metrics(golden_df["gold_should_escalate"].tolist(), maj_pred_escalate),
    }

    # ---- Baseline 2: TF-IDF + LogReg ----
    tfidf = TfidfBaseline().fit(
        knowledge_df["customer_message_clean"], knowledge_df["intent"], knowledge_df["support_response_clean"]
    )
    tfidf_pred_intents, tfidf_pred_escalate = [], []
    for msg in golden_df["customer_message"]:
        pred = tfidf.predict_intent(msg)
        tfidf_pred_intents.append(pred["intent"])
        hits = tfidf.retrieve(msg, top_k=1)
        top_sim = hits[0]["similarity"] if hits else 0.0
        esc = tfidf.predict_escalation(top_sim)
        tfidf_pred_escalate.append(esc["decision"] == "ESCALATE")

    results["tfidf_baseline"] = {
        "intent": intent_metrics(golden_df["gold_intent"].tolist(), tfidf_pred_intents),
        "escalation": escalation_metrics(golden_df["gold_should_escalate"].tolist(), tfidf_pred_escalate),
    }

    # ---- Rule-based escalation policy applied on top of TF-IDF signals (preview of full system's escalation logic) ----
    rule_pred_escalate = []
    for msg, intent_pred in zip(golden_df["customer_message"], tfidf_pred_intents):
        hits = tfidf.retrieve(msg, top_k=1)
        top_sim = hits[0]["similarity"] if hits else None
        d = decide(msg, intent_pred, 1.0, top_sim, len(hits))
        rule_pred_escalate.append(d.decision == "ESCALATE")
    results["rule_escalation_policy_on_tfidf"] = {
        "escalation": escalation_metrics(golden_df["gold_should_escalate"].tolist(), rule_pred_escalate),
    }

    out_path = RESULTS_DIR / "baseline_metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n=== Majority baseline ===")
    print(f"Intent accuracy: {results['majority_baseline']['intent']['accuracy']:.3f}, "
          f"macro-F1: {results['majority_baseline']['intent']['macro_f1']:.3f}")
    print(f"Escalation: {results['majority_baseline']['escalation']}")

    print(f"\n=== TF-IDF baseline ===")
    print(f"Intent accuracy: {results['tfidf_baseline']['intent']['accuracy']:.3f}, "
          f"macro-F1: {results['tfidf_baseline']['intent']['macro_f1']:.3f}")
    print(f"Escalation: {results['tfidf_baseline']['escalation']}")

    print(f"\n=== Rule escalation policy (on TF-IDF retrieval) ===")
    print(f"Escalation: {results['rule_escalation_policy_on_tfidf']['escalation']}")

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
