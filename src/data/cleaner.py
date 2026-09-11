"""Filters out pairs that are unusable for training/eval/retrieval."""
from __future__ import annotations

import re

import pandas as pd

URL_RE = re.compile(r"https?://\S+")
HANDLE_RE = re.compile(r"@\w+")


def clean_text(text: str) -> str:
    text = URL_RE.sub("", str(text))
    text = HANDLE_RE.sub("", text)
    return " ".join(text.split()).strip()


def filter_pairs(pairs_df: pd.DataFrame, min_chars: int = 8, max_chars: int = 500) -> pd.DataFrame:
    """
    Drop pairs where either side is empty/too short/too long after
    cleaning. Returns a NEW dataframe with cleaned text columns; does not
    mutate the input (which itself must never mutate the raw source).
    """
    df = pairs_df.copy()
    df["customer_message_clean"] = df["customer_message"].map(clean_text)
    df["support_response_clean"] = df["support_response"].map(clean_text)

    before = len(df)
    df = df[
        df["customer_message_clean"].str.len().between(min_chars, max_chars)
        & df["support_response_clean"].str.len().between(min_chars, max_chars)
    ].reset_index(drop=True)
    dropped = before - len(df)
    df.attrs["n_dropped"] = dropped
    df.attrs["n_before"] = before
    return df
