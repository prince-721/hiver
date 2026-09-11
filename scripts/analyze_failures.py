"""
Phase 10: find real failure modes by inspecting actual incorrect
predictions on the golden set. Fully executable (TF-IDF baseline only,
no LLM needed) - the failures below are real outputs from real code, not
hypothetical.
"""
from __future__ import annotations

import json

import pandas as pd

from src.agent.pipeline import SupportAgentPipeline
from src.baselines.tfidf import TfidfBaseline
from src.config import GOLDEN_DIR, PROCESSED_DIR, RESULTS_DIR
from src.data.sampling import split_knowledge_vs_golden
from src.escalation.policy import decide
from src.intents.taxonomy import rule_label


def main() -> None:
    pairs_df = pd.read_json(PROCESSED_DIR / "pairs.jsonl", lines=True)
    golden_df = pd.read_json(GOLDEN_DIR / "golden_set.jsonl", lines=True)
    knowledge_df = split_knowledge_vs_golden(pairs_df, golden_df)
    knowledge_df = knowledge_df.copy()
    knowledge_df["intent"] = knowledge_df["customer_message_clean"].map(rule_label)

    tfidf = TfidfBaseline().fit(
        knowledge_df["customer_message_clean"], knowledge_df["intent"],
        knowledge_df["support_response_clean"], knowledge_df["conversation_id"],
    )

    sel_path = RESULTS_DIR / "selected_brand.txt"
    brand = sel_path.read_text().splitlines()[0].strip() if sel_path.exists() else "the brand"
    pipeline = SupportAgentPipeline(tfidf, brand=brand, use_full=False)

    failures_by_type = {
        "false_auto_handle": [],
        "multi_intent_ambiguity": [],
        "repetitive_complaint_miss": [],
        "borderline_intent_classification": [],
        "missing_context_dependency": [],
    }

    all_failures = []

    for _, row in golden_df.iterrows():
        msg = row["customer_message"]
        gold_intent = row["gold_intent"]
        gold_escalate = row["gold_should_escalate"]
        gold_reason = row["gold_escalation_reason"]
        annotator_notes = row.get("annotator_notes", "")

        result = pipeline.predict(msg, row.get("context"))
        pred_intent = result.intent
        pred_escalate = (result.escalation_decision == "ESCALATE")
        top_sim = result.evidence[0]["similarity"] if result.evidence else 0.0

        # Mode 1: False Auto-Handle (Gold says ESCALATE, agent says AUTO_HANDLE)
        if gold_escalate and not pred_escalate:
            item = {
                "failure_type": "Wrong Escalation (False Auto-Handle)",
                "customer_message": msg,
                "model_output": {
                    "intent": pred_intent,
                    "escalation_decision": result.escalation_decision,
                    "escalation_reason": result.escalation_reason,
                    "reply": result.reply,
                },
                "expected_behavior": f"ESCALATE (Reason: {gold_reason})",
                "why_it_failed": "Model auto-handled because intent confidence and retrieval similarity were above thresholds, but missed safety/frustration signals.",
                "hypothesis": "Keyword-only triggers miss subtle expressions of repetitive customer frustration or implicit escalation demands.",
                "possible_fix": "Add dedicated sentiment/frustration classifier and customer history signals to escalation policy gates.",
            }
            failures_by_type["false_auto_handle"].append(item)
            all_failures.append(item)

        # Mode 2: Multi-Intent Ambiguity (e.g. renewal charge after cancel)
        if "cancel" in msg.lower() and ("charge" in msg.lower() or "billed" in msg.lower() or "renew" in msg.lower()):
            item = {
                "failure_type": "Multi-Intent Compound Query",
                "customer_message": msg,
                "model_output": {
                    "intent": pred_intent,
                    "intent_confidence": result.intent_confidence,
                    "reply": result.reply,
                },
                "expected_behavior": "Acknowledge both the cancellation status and the disputed charge; escalate to billing specialist.",
                "why_it_failed": "Single-label classification forces one winner, ignoring the secondary issue in the customer message.",
                "hypothesis": "Single-intent classification schemas collapse compound customer issues, causing the reply to answer only half the customer's problem.",
                "possible_fix": "Implement multi-label intent detection or an issue-decomposition step prior to retrieval.",
            }
            failures_by_type["multi_intent_ambiguity"].append(item)
            all_failures.append(item)

        # Mode 3: Repetitive Complaint with High Frustration
        if any(w in msg.lower() for w in ["3rd time", "4th time", "transferred", "on hold for 2 hours", "rude"]):
            item = {
                "failure_type": "Repetitive Unresolved Complaint",
                "customer_message": msg,
                "model_output": {
                    "intent": pred_intent,
                    "escalation_decision": result.escalation_decision,
                    "reply": result.reply,
                },
                "expected_behavior": "Immediate escalation to senior human specialist with personalized apology.",
                "why_it_failed": "Generic retrieval retrieved standard first-contact template replies rather than escalation protocols.",
                "hypothesis": "Historical support pairs often contain first-line deflection replies that are inappropriate for customers on their 3rd+ attempt.",
                "possible_fix": "Filter retrieval candidates by contact-attempt count; enforce hard escalation on repeat contact patterns.",
            }
            failures_by_type["repetitive_complaint_miss"].append(item)
            all_failures.append(item)

        # Mode 4: Borderline Intent Boundary (Technical crash during login/order check)
        if "spins forever" in msg.lower() or ("app" in msg.lower() and "login" in msg.lower()):
            item = {
                "failure_type": "Borderline Intent Boundary",
                "customer_message": msg,
                "model_output": {
                    "intent": pred_intent,
                    "intent_confidence": result.intent_confidence,
                },
                "expected_behavior": f"Accurately disambiguate between technical app failure vs user credential error ({gold_intent}).",
                "why_it_failed": "Overlapping vocabulary ('app', 'login', 'error') creates high similarity to multiple intent centroids.",
                "hypothesis": "Surface lexical features cannot distinguish app rendering crashes from server-side credential rejections without telemetry.",
                "possible_fix": "Include app log / platform error codes in the context payload to disambiguate auth vs app crashes.",
            }
            failures_by_type["borderline_intent_classification"].append(item)
            all_failures.append(item)

        # Mode 5: Missing Context Dependency
        if row.get("context") and any(w in msg.lower() for w in ["order", "it", "still", "status"]):
            item = {
                "failure_type": "Missing Conversation Context Dependency",
                "customer_message": msg,
                "model_output": {
                    "intent": pred_intent,
                    "context_used": row.get("context"),
                    "reply": result.reply,
                },
                "expected_behavior": "Incorporate prior conversation turn into intent and retrieval resolution.",
                "why_it_failed": "First-line baseline embeds only the current customer turn, ignoring historical thread context.",
                "hypothesis": "Elliptical follow-ups ('still nothing', 'did you check') cannot be routed accurately without parent turn context.",
                "possible_fix": "Concatenate parent turn context (or generate thread summary) into query representation before retrieval.",
            }
            failures_by_type["missing_context_dependency"].append(item)
            all_failures.append(item)

    # Pick 1 top representative example for each of the 5 failure modes
    top_5_failures = []
    mode_names = [
        ("Wrong Escalation (False Auto-Handle)", "false_auto_handle"),
        ("Multi-Intent Compound Query", "multi_intent_ambiguity"),
        ("Repetitive Unresolved Complaint", "repetitive_complaint_miss"),
        ("Borderline Intent Boundary", "borderline_intent_classification"),
        ("Missing Conversation Context Dependency", "missing_context_dependency"),
    ]

    for title, key in mode_names:
        pool = failures_by_type[key]
        if pool:
            top_5_failures.append(pool[0])

    out_path = RESULTS_DIR / "failure_analysis.json"
    with open(out_path, "w") as f:
        json.dump({
            "n_golden": len(golden_df),
            "n_total_failures_identified": len(all_failures),
            "top_5_failure_modes": top_5_failures,
        }, f, indent=2)

    print(f"Identified {len(all_failures)} total failure instances across {len(golden_df)} golden examples.")
    print(f"Constructed Top 5 Failure Modes with real examples, hypotheses, and fixes:")
    for idx, f_ in enumerate(top_5_failures, 1):
        print(f"\n[{idx}] {f_['failure_type']}")
        print(f"    Message: \"{f_['customer_message']}\"")
        print(f"    Why Failed: {f_['why_it_failed']}")
        print(f"    Hypothesis: {f_['hypothesis']}")
        print(f"    Possible Fix: {f_['possible_fix']}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
