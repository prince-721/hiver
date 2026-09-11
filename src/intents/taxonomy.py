"""
Keyword-rule labeler grounded directly in the labeling_rules written into
data/intent_taxonomy.json. Used to (a) produce first-pass gold labels that
a human reviewer then spot-checks/corrects, and (b) as a transparent,
inspectable classifier baseline component.

This is NOT a replacement for human labeling of the real golden set - see
DECISIONS.md for why a rule-first pass plus human review was chosen over
labeling 200 examples from scratch by hand with no scaffolding.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from src.config import DATA_DIR

_RULES: dict[str, list[str]] = {
    "login_problem": [r"log(ging)? ?in", r"password", r"locked out", r"account access"],
    "refund_request": [r"refund"],
    "billing_issue": [r"charg(ed|e)", r"billed", r"renew(ed|al)", r"plan is"],
    "delivery_problem": [r"deliver", r"transit", r"tracking", r"package", r"never received"],
    "technical_problem": [r"crash", r"error", r"won'?t load", r"bug", r"app "],
    "cancellation": [r"cancel"],
    "complaint": [r"disappointed", r"3rd time|third time", r"no fix|no reply|still nothing"],
}


def load_taxonomy() -> dict:
    path = Path(DATA_DIR) / "intent_taxonomy.json"
    return json.loads(path.read_text())


def rule_label(text: str) -> str:
    """Returns the first matching intent by rule priority, else 'other'."""
    t = text.lower()
    for intent, patterns in _RULES.items():
        for p in patterns:
            if re.search(p, t):
                return intent
    return "other"
