"""
Phase 1: inspect the dataset and select a brand by measurable criteria.

Usage:
    python -m scripts.inspect_dataset --data-source mock
    python -m scripts.inspect_dataset --data-source kaggle --top-n 15

Writes results/brand_selection.csv (the table required by STEP 2 of the
assignment) and prints the selected brand + reason.
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.config import RESULTS_DIR
from src.data.loader import identify_brands, load_raw, missing_value_report
from src.data.threads import reconstruct_pairs


def score_brands(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    candidates = identify_brands(df).head(top_n)
    rows = []
    for brand, n_support_msgs in candidates.items():
        pairs_df, stats = reconstruct_pairs(df, brand)
        avg_len = pairs_df["customer_message"].str.len().mean() if len(pairs_df) else 0
        rows.append({
            "brand": brand,
            "support_messages": n_support_msgs,
            "usable_pairs": stats.usable_pairs,
            "pct_customer_msgs_answered": stats.pct_answered,
            "avg_customer_msg_len_chars": round(avg_len, 1),
        })
    table = pd.DataFrame(rows).sort_values("usable_pairs", ascending=False).reset_index(drop=True)
    return table


def select_brand(table: pd.DataFrame, min_pairs: int = 50) -> tuple[str, str]:
    eligible = table[table["usable_pairs"] >= min_pairs]
    pool = eligible if len(eligible) else table
    best = pool.iloc[0]
    reason = (
        f"Selected '{best['brand']}': highest usable customer/support pair count "
        f"({int(best['usable_pairs'])}) among candidates with >= {min_pairs} pairs, "
        f"with {best['pct_customer_msgs_answered']}% of customer messages answered."
    )
    return best["brand"], reason


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-source", default="mock", choices=["mock", "kaggle"])
    ap.add_argument("--top-n", type=int, default=10, help="how many top support accounts to score")
    ap.add_argument("--min-pairs", type=int, default=50)
    args = ap.parse_args()

    df = load_raw(args.data_source)
    print(f"Loaded {len(df)} rows.")
    print("\nMissing-value report:")
    print(missing_value_report(df).to_string(index=False))

    table = score_brands(df, args.top_n)
    print("\nBrand candidate table:")
    print(table.to_string(index=False))

    out_path = RESULTS_DIR / "brand_selection.csv"
    table.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")

    brand, reason = select_brand(table, min_pairs=args.min_pairs)
    print(f"\nSELECTED BRAND: {brand}")
    print(f"REASON: {reason}")

    with open(RESULTS_DIR / "selected_brand.txt", "w") as f:
        f.write(f"{brand}\n{reason}\n")


if __name__ == "__main__":
    main()
