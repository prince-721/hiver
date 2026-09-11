"""
Phase 2: reconstruct conversation pairs for the selected brand, clean them,
and write data/processed/pairs.jsonl (only derived data - never the raw
dataset - is stored in the repo, per the assignment's instructions).

Usage:
    python -m scripts.build_sample --data-source mock
    python -m scripts.build_sample --data-source kaggle --brand "@AmazonHelp" --sample-size 20000
"""
from __future__ import annotations

import argparse

from src.config import PROCESSED_DIR, RESULTS_DIR, SETTINGS
from src.data.cleaner import filter_pairs
from src.data.loader import load_raw
from src.data.threads import reconstruct_pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-source", default="mock", choices=["mock", "kaggle"])
    ap.add_argument("--brand", default=None, help="defaults to results/selected_brand.txt from inspect_dataset.py")
    ap.add_argument("--sample-size", type=int, default=SETTINGS.sample_size)
    ap.add_argument("--seed", type=int, default=SETTINGS.random_seed)
    args = ap.parse_args()

    brand = args.brand
    if brand is None:
        sel_path = RESULTS_DIR / "selected_brand.txt"
        if not sel_path.exists():
            raise SystemExit("No --brand given and no results/selected_brand.txt found - run inspect_dataset.py first.")
        brand = sel_path.read_text().splitlines()[0].strip()
        print(f"Using previously selected brand: {brand}")

    df = load_raw(args.data_source)
    if len(df) > args.sample_size:
        df = df.sample(n=args.sample_size, random_state=args.seed)

    pairs_df, stats = reconstruct_pairs(df, brand)
    print(f"Reconstructed {stats.usable_pairs} usable pairs "
          f"({stats.pct_answered}% of {stats.total_inbound} inbound msgs answered by this brand).")

    cleaned = filter_pairs(pairs_df)
    print(f"After cleaning: {len(cleaned)} pairs kept, {cleaned.attrs['n_dropped']} dropped "
          f"(too short/long/empty).")

    out_path = PROCESSED_DIR / "pairs.jsonl"
    cleaned.to_json(out_path, orient="records", lines=True)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
