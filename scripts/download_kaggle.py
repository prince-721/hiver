"""
Download the Kaggle Customer Support on Twitter dataset using kagglehub
and copy twcs.csv into data/raw/twcs.csv for the support agent pipeline.

Usage:
    py -3.13 scripts/download_kaggle.py
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
TARGET_CSV = RAW_DIR / "twcs.csv"


def download_dataset() -> Path:
    try:
        import kagglehub
    except ImportError:
        raise SystemExit(
            "kagglehub is not installed. Please run:\n"
            "    py -3.13 -m pip install kagglehub"
        )

    print("Downloading dataset 'thoughtvector/customer-support-on-twitter' via kagglehub...")
    dataset_path = Path(kagglehub.dataset_download("thoughtvector/customer-support-on-twitter"))
    print(f"Dataset downloaded to cache: {dataset_path}")

    # Locate twcs.csv in downloaded files
    candidates = list(dataset_path.glob("**/twcs.csv"))
    if not candidates:
        # Fallback to any csv
        candidates = list(dataset_path.glob("**/*.csv"))

    if not candidates:
        raise FileNotFoundError(f"Could not find twcs.csv in {dataset_path}")

    source_csv = candidates[0]
    print(f"Found source CSV: {source_csv} ({source_csv.stat().st_size / 1e6:.1f} MB)")

    if TARGET_CSV.exists():
        print(f"Target already exists at {TARGET_CSV}, overwriting...")

    print(f"Copying to {TARGET_CSV}...")
    shutil.copy2(source_csv, TARGET_CSV)
    print(f"Successfully placed dataset at: {TARGET_CSV}")
    print("\nYou can now run the pipeline on real Kaggle data:")
    print("    py -3.13 -m scripts.inspect_dataset --data-source kaggle --top-n 15")
    print("    py -3.13 -m scripts.build_sample --data-source kaggle --sample-size 20000")
    print("    py -3.13 -m scripts.evaluate --data-source kaggle --full --limit 10")
    return TARGET_CSV


if __name__ == "__main__":
    download_dataset()
