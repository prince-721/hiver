"""
Baseline 1 (trivial): the minimum reference point every real system must
beat.

- Intent: always predict the single most frequent intent in the training
  pairs (majority class).
- Reply: always return one fixed generic response.
- Escalation: always AUTO_HANDLE (documented choice - see DECISIONS.md;
  this makes the baseline's automation-coverage look great and its
  correctness/safety look terrible, which is exactly the point: it shows
  why coverage alone is a misleading headline metric).
"""
from __future__ import annotations

import pandas as pd

GENERIC_REPLY = "Thanks for reaching out! Please DM us your order/account details and we'll look into this."


class MajorityBaseline:
    def __init__(self) -> None:
        self.majority_intent: str | None = None

    def fit(self, train_intents: pd.Series) -> "MajorityBaseline":
        self.majority_intent = train_intents.mode().iloc[0]
        return self

    def predict_intent(self, message: str) -> dict:
        if self.majority_intent is None:
            raise RuntimeError("Call fit() first.")
        return {"intent": self.majority_intent, "confidence": None, "top_candidates": [self.majority_intent]}

    def predict_reply(self, message: str) -> str:
        return GENERIC_REPLY

    def predict_escalation(self) -> dict:
        return {"decision": "AUTO_HANDLE", "reason": "Trivial baseline always auto-handles.", "confidence": None}
