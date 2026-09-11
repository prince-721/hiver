"""
High-level retrieval interface used by the reply generator.

NOT EXECUTED END-TO-END IN THIS SANDBOX (depends on Embedder/VectorIndex,
see those modules' docstrings). The TF-IDF baseline in
src/baselines/tfidf.py is the fully-executed stand-in for this component
during development here.
"""
from __future__ import annotations

import pandas as pd

from src.config import SETTINGS
from src.retrieval.embeddings import Embedder
from src.retrieval.index import VectorIndex


class HistoricalRetriever:
    def __init__(self, embedder: Embedder | None = None) -> None:
        self.embedder = embedder or Embedder()
        self.index: VectorIndex | None = None

    def build(self, knowledge_df: pd.DataFrame) -> None:
        """knowledge_df must already have golden-set rows excluded (see
        src/data/sampling.split_knowledge_vs_golden)."""
        texts = knowledge_df["customer_message_clean"].tolist()
        vectors = self.embedder.encode(texts)
        self.index = VectorIndex(dim=vectors.shape[1])
        metadata = [
            {
                "conversation_id": row["conversation_id"],
                "customer_message": row["customer_message_clean"],
                "historical_response": row["support_response_clean"],
            }
            for _, row in knowledge_df.iterrows()
        ]
        self.index.add(vectors, metadata)

    def retrieve(self, message: str, top_k: int | None = None) -> list[dict]:
        if self.index is None:
            raise RuntimeError("Call build() or load an index first.")
        top_k = top_k or SETTINGS.retrieval_top_k
        query_vec = self.embedder.encode([message])[0]
        return self.index.search(query_vec, top_k=top_k)
