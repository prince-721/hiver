"""
Phase 6/7 glue: build and persist the FAISS retrieval index from the
knowledge pool (golden set excluded).

NOT EXECUTED IN THIS SANDBOX - requires sentence-transformers + faiss-cpu,
which need network access to install/download the embedding model. Run
locally after `pip install -r requirements.txt`:

    python -m scripts.build_index --data-source kaggle

Writes results/index/knowledge (knowledge.faiss + knowledge.meta.json via
src/retrieval/index.py's save()).
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.config import GOLDEN_DIR, PROCESSED_DIR, RESULTS_DIR
from src.data.sampling import split_knowledge_vs_golden
from src.retrieval.retriever import HistoricalRetriever


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-source", default="mock", choices=["mock", "kaggle"])
    args = ap.parse_args()

    pairs_df = pd.read_json(PROCESSED_DIR / "pairs.jsonl", lines=True)
    golden_path = GOLDEN_DIR / "golden_set.jsonl"
    if golden_path.exists():
        golden_df = pd.read_json(golden_path, lines=True)
        knowledge_df = split_knowledge_vs_golden(pairs_df, golden_df)
        print(f"Excluded {knowledge_df.attrs.get('n_removed_for_golden', 0)} golden-set rows from the index.")
    else:
        knowledge_df = pairs_df
        print("No golden set found - indexing all pairs (fine for a first smoke test only).")

    retriever = HistoricalRetriever()
    retriever.build(knowledge_df)

    out_dir = RESULTS_DIR / "index"
    out_dir.mkdir(exist_ok=True)
    retriever.index.save(str(out_dir / "knowledge"))
    print(f"Indexed {len(knowledge_df)} historical cases -> {out_dir}/knowledge.faiss")


if __name__ == "__main__":
    main()
