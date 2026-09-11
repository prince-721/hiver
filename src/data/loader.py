"""
Loads the Customer Support on Twitter dataset (or the mock sample) and
validates it against the expected schema. Never mutates the source file.

Real dataset setup (not done automatically - no network access from this
pipeline by design, see README):
    1. Download from https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
    2. Place twcs.csv at data/raw/twcs.csv
    3. Set SUPPORT_AGENT_DATA_SOURCE=kaggle
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import MOCK_RAW_DIR, RAW_DIR, SETTINGS

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = {
    "tweet_id", "author_id", "inbound", "created_at", "text",
    "response_tweet_id", "in_response_to_tweet_id",
}


class SchemaError(ValueError):
    pass


def load_raw(data_source: str | None = None) -> pd.DataFrame:
    """Load the raw tweet-level dataframe (mock or real) and validate columns."""
    source = data_source or SETTINGS.data_source
    if source == "mock":
        path = Path(MOCK_RAW_DIR) / "twcs_mock.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found - run `python scripts/generate_mock_data.py` first."
            )
    elif source == "kaggle":
        path = Path(RAW_DIR) / SETTINGS.raw_csv_name
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Download twcs.csv from the Kaggle dataset "
                "(thoughtvector/customer-support-on-twitter) and place it there."
            )
    else:
        raise ValueError(f"Unknown data_source: {source}")

    df = pd.read_csv(path)
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise SchemaError(
            f"Dataset at {path} is missing expected columns: {missing}. "
            "If the real Kaggle file uses different column names, adapt "
            "EXPECTED_COLUMNS and the loader rather than assuming this schema."
        )

    # normalize dtypes
    df["inbound"] = df["inbound"].astype(bool)
    for col in ("response_tweet_id", "in_response_to_tweet_id"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info("Loaded %d rows from %s", len(df), path)
    return df


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Simple missing-value inventory, used by scripts/inspect_dataset.py."""
    report = pd.DataFrame({
        "column": df.columns,
        "n_missing": [df[c].isna().sum() for c in df.columns],
        "pct_missing": [round(100 * df[c].isna().mean(), 2) for c in df.columns],
    })
    return report.sort_values("n_missing", ascending=False).reset_index(drop=True)


def identify_brands(df: pd.DataFrame) -> pd.Series:
    """
    Brand accounts are the non-inbound (support) authors. Returns a
    value_counts Series of support-account author_ids, i.e. candidate
    brands, ranked by number of outbound (support) messages.
    """
    support_msgs = df[~df["inbound"]]
    return support_msgs["author_id"].value_counts()
