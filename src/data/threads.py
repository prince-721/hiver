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
    given brand account. Also returns simple thread statistics used for
    brand selection.
    """
    by_id = df.set_index("tweet_id")
    brand_msgs = df[(df["author_id"] == brand_author_id) & (~df["inbound"])]

    records = []
    for _, brand_row in brand_msgs.iterrows():
        parent_id = brand_row["in_response_to_tweet_id"]
        if pd.isna(parent_id) or parent_id not in by_id.index:
            continue
        parent = by_id.loc[parent_id]
        if not parent["inbound"]:
            continue  # brand replying to itself/another brand tweet - not a customer pair
        records.append({
            "conversation_id": f"{int(parent_id)}_{int(brand_row['tweet_id'])}",
            "customer_message": parent["text"],
            "support_response": brand_row["text"],
            "brand": brand_author_id,
            "customer_tweet_id": int(parent_id),
            "support_tweet_id": int(brand_row["tweet_id"]),
            "timestamp": brand_row.get("created_at"),
        })

    pairs_df = pd.DataFrame.from_records(records)

    total_inbound = int((df["inbound"]).sum())
    # inbound tweets that were ever answered by *this* brand
    answered_ids = set(pairs_df["customer_tweet_id"]) if len(pairs_df) else set()
    unanswered = total_inbound - len(answered_ids)
    stats = ThreadStats(
        total_inbound=total_inbound,
        usable_pairs=len(pairs_df),
        unanswered_inbound=unanswered,
        pct_answered=round(100 * len(pairs_df) / max(total_inbound, 1), 2),
    )
    return pairs_df, stats
