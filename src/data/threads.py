"""
Reconstructs customer_message -> support_response pairs from the raw
tweet-level table using the response_tweet_id / in_response_to_tweet_id
links provided by the dataset.

A "usable pair" is: an inbound (customer) tweet whose response_tweet_id
points at an outbound (support) tweet authored by the target brand.
Anything else (customer tweets nobody replied to, support tweets replying
to other support tweets, orphaned rows) is dropped and counted, not
silently discarded.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class ThreadStats:
    total_inbound: int
    usable_pairs: int
    unanswered_inbound: int
    pct_answered: float


def reconstruct_pairs(df: pd.DataFrame, brand_author_id: str) -> tuple[pd.DataFrame, ThreadStats]:
    """
    Build one row per (customer_message, support_response) pair for a
    given brand account using vectorized merge. Also returns thread statistics.
    """
    brand_msgs = df[(df["author_id"] == brand_author_id) & (~df["inbound"])].dropna(subset=["in_response_to_tweet_id"])
    inbound_msgs = df[df["inbound"]]

    merged = brand_msgs.merge(
        inbound_msgs,
        left_on="in_response_to_tweet_id",
        right_on="tweet_id",
        suffixes=("_support", "_customer"),
    )

    pairs_df = pd.DataFrame({
        "conversation_id": merged["in_response_to_tweet_id_support"].astype(int).astype(str) + "_" + merged["tweet_id_support"].astype(int).astype(str),
        "customer_message": merged["text_customer"],
        "support_response": merged["text_support"],
        "brand": brand_author_id,
        "customer_tweet_id": merged["in_response_to_tweet_id_support"].astype(int),
        "support_tweet_id": merged["tweet_id_support"].astype(int),
        "timestamp": merged["created_at_support"],
    })

    total_inbound = int((df["inbound"]).sum())
    answered_ids = set(pairs_df["customer_tweet_id"]) if len(pairs_df) else set()
    unanswered = total_inbound - len(answered_ids)
    stats = ThreadStats(
        total_inbound=total_inbound,
        usable_pairs=len(pairs_df),
        unanswered_inbound=unanswered,
        pct_answered=round(100 * len(pairs_df) / max(total_inbound, 1), 2),
    )
    return pairs_df, stats
