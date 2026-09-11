"""
Phase 3: run TF-IDF + KMeans clustering over the brand's customer
messages and print top terms per cluster, to support hand-naming the
final intent taxonomy (data/intent_taxonomy.json).

This script does NOT write the taxonomy automatically - intent naming
is a human judgment call the assignment explicitly asks for, done by
inspecting these clusters.
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.config import PROCESSED_DIR
from src.intents.discovery import discover_clusters


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-clusters", type=int, default=8)
    args = ap.parse_args()

    df = pd.read_json(PROCESSED_DIR / "pairs.jsonl", lines=True)
    labels, top_terms = discover_clusters(df["customer_message_clean"], n_clusters=args.n_clusters)
    df["cluster"] = labels

    for c, terms in sorted(top_terms.items()):
        n = (df["cluster"] == c).sum()
        print(f"\nCluster {c} (n={n}): {', '.join(terms)}")
        for ex in df[df["cluster"] == c]["customer_message_clean"].head(3):
            print(f"    - {ex}")


if __name__ == "__main__":
    main()
