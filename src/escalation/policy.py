"""
Escalation decision: AUTO_HANDLE vs ESCALATE, with a stated reason.

Deliberately rule-based and inspectable (not an LLM call) so that the
safety-critical decision is auditable. See DECISIONS.md for why.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from src.config import SETTINGS

HIGH_RISK_PATTERNS = [
    r"\bhack(ed)?\b", r"\bfraud\b", r"\blawsuit\b", r"\bsue\b", r"\blegal\b",
    r"\bunauthoriz", r"\bstolen\b", r"\bdata breach\b",
]
HUMAN_REQUEST_PATTERNS = [r"talk to a (human|person|agent)", r"real person", r"speak to someone"]
REPEATED_COMPLAINT_PATTERNS = [r"\b(2nd|3rd|third|second|again)\b.*(time|contact|written)"]


@dataclass
class EscalationDecision:
    decision: str  # "AUTO_HANDLE" | "ESCALATE"
    reason: str
    confidence: float


def decide(
    message: str,
    intent: str,
    intent_confidence: float,
    top_retrieval_similarity: float | None,
    n_relevant_cases: int,
) -> EscalationDecision:
    text = message.lower()

    for pat in HIGH_RISK_PATTERNS:
        if re.search(pat, text):
            return EscalationDecision("ESCALATE", f"High-risk signal detected ({pat}); requires human review.", 0.95)

    for pat in HUMAN_REQUEST_PATTERNS:
        if re.search(pat, text):
            return EscalationDecision("ESCALATE", "Customer explicitly asked for a human.", 0.99)

    if intent == "complaint" or any(re.search(p, text) for p in REPEATED_COMPLAINT_PATTERNS):
        return EscalationDecision("ESCALATE", "Repeated/unresolved complaint pattern; low trust in auto-reply resolving it.", 0.8)

    if n_relevant_cases == 0 or top_retrieval_similarity is None:
        return EscalationDecision("ESCALATE", "No relevant historical resolution found to ground a reply.", 0.85)

    if top_retrieval_similarity < SETTINGS.min_retrieval_similarity:
        return EscalationDecision(
            "ESCALATE",
            f"Best retrieved similarity ({top_retrieval_similarity:.2f}) below threshold "
            f"({SETTINGS.min_retrieval_similarity}); insufficient grounding evidence.",
            0.75,
        )

    if intent_confidence < SETTINGS.min_intent_confidence:
        return EscalationDecision(
            "ESCALATE",
            f"Intent classifier confidence ({intent_confidence:.2f}) below threshold "
            f"({SETTINGS.min_intent_confidence}).",
            0.7,
        )

    if intent == "billing_issue":
        # account/money-specific issues are treated conservatively even when
        # confidence/retrieval look fine, since acting on them wrongly is costly.
        return EscalationDecision(
            "ESCALATE",
            "Billing disputes involve account-specific financial actions we choose not to auto-resolve.",
            0.7,
        )

    return EscalationDecision(
        "AUTO_HANDLE",
        f"High-confidence intent ({intent_confidence:.2f}), strong retrieval grounding "
        f"({top_retrieval_similarity:.2f}), no high-risk signals.",
        round((intent_confidence + top_retrieval_similarity) / 2, 2),
    )
