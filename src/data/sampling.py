"""
Splits cleaned pairs into:
  - knowledge_df: used for retrieval index + baseline training
  - golden_df: held out, loaded separately from data/golden/golden_set.jsonl

The split is done by conversation_id (a stable identifier tying a specific
customer tweet to a specific support reply), NOT by message text. Text
matching was tried first and rejected - see DECISIONS.md: near-duplicate
phrasing across different customers (common in templated/recurring
support issues) means text-based exclusion can remove far more of the
knowledge pool than intended, or too little on real varied-phrasing data.
ID-based exclusion is exact regardless of phrasing overlap.
"""
from __future__ import annotations

import pandas as pd


def split_knowledge_vs_golden(pairs_df: pd.DataFrame, golden_df: pd.DataFrame) -> pd.DataFrame:
    if "conversation_id" in golden_df.columns:
        golden_ids = set(golden_df["conversation_id"])
        before = len(pairs_df)
        knowledge_df = pairs_df[~pairs_df["conversation_id"].isin(golden_ids)].reset_index(drop=True)
    else:
        # fallback for golden sets built before conversation_id was tracked
        golden_messages = set(golden_df["customer_message"])
        before = len(pairs_df)
        knowledge_df = pairs_df[~pairs_df["customer_message_clean"].isin(golden_messages)].reset_index(drop=True)
    removed = before - len(knowledge_df)
    knowledge_df.attrs["n_removed_for_golden"] = removed
    return knowledge_df
