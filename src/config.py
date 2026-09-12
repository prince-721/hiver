"""
Central configuration for the Hiver support-agent pipeline.

Everything that should be changeable (brand, sample size, seed, model name,
paths) lives here and can be overridden by environment variables or CLI
flags in the scripts. Nothing here is a secret — the API key is read from
the environment only, never hardcoded.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_env_file() -> None:
    env_path = ROOT / ".env"
    if env_path.exists():
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k and k not in os.environ:
                        os.environ[k] = v
        except Exception:
            pass


_load_env_file()

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MOCK_RAW_DIR = DATA_DIR / "mock_raw"
PROCESSED_DIR = DATA_DIR / "processed"
GOLDEN_DIR = DATA_DIR / "golden"
RESULTS_DIR = ROOT / "results"
CACHE_DIR = ROOT / ".cache"

for d in (RAW_DIR, MOCK_RAW_DIR, PROCESSED_DIR, GOLDEN_DIR, RESULTS_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)


@dataclass
class Settings:
    # --- data source ---
    data_source: str = os.environ.get(
        "SUPPORT_AGENT_DATA_SOURCE",
        "kaggle" if (RAW_DIR / "twcs.csv").exists() else "mock"
    )
    raw_csv_name: str = "twcs.csv"

    # --- brand / sampling ---
    brand: str | None = os.environ.get("SUPPORT_AGENT_BRAND")  # None = auto-select
    sample_size: int = int(os.environ.get("SUPPORT_AGENT_SAMPLE_SIZE", "20000"))
    random_seed: int = int(os.environ.get("SUPPORT_AGENT_SEED", "42"))

    # --- golden set ---
    golden_set_size: int = int(os.environ.get("SUPPORT_AGENT_GOLDEN_SIZE", "200"))

    # --- retrieval ---
    embedding_model: str = os.environ.get(
        "SUPPORT_AGENT_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    retrieval_top_k: int = int(os.environ.get("SUPPORT_AGENT_TOP_K", "5"))

    # --- LLM Provider (Groq is free; Anthropic / OpenAI also supported) ---
    groq_api_key: str | None = os.environ.get("GROQ_API_KEY")
    anthropic_api_key: str | None = os.environ.get("ANTHROPIC_API_KEY")
    llm_model: str = os.environ.get(
        "SUPPORT_AGENT_LLM_MODEL",
        "openai/gpt-oss-120b" if os.environ.get("GROQ_API_KEY") else "claude-sonnet-4-6",
    )
    llm_provider_enabled: bool = bool(os.environ.get("GROQ_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))

    # --- escalation thresholds (documented + justified in DECISIONS.md) ---
    min_intent_confidence: float = float(os.environ.get("ESCALATE_MIN_INTENT_CONF", "0.55"))
    min_retrieval_similarity: float = float(os.environ.get("ESCALATE_MIN_SIM", "0.45"))


SETTINGS = Settings()
