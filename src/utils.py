"""Shared utilities for the support agent pipeline."""
from __future__ import annotations

import json
import logging
import time
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)


def groq_api_call(
    api_key: str,
    model: str,
    messages: list[dict],
    max_tokens: int = 400,
    temperature: float = 0.2,
    max_retries: int = 8,
    base_delay: float = 2.0,
) -> dict:
    """Make a Groq API call with exponential backoff retry on 429/5xx errors."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    for attempt in range(max_retries):
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "HiverSupportAgent/1.0",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                retry_header = e.headers.get("retry-after") or e.headers.get("x-ratelimit-reset-tokens")
                if retry_header:
                    try:
                        val_str = retry_header.rstrip("s").strip()
                        delay = max(delay, float(val_str) + 0.5)
                    except (ValueError, TypeError):
                        pass
                logger.warning(
                    "Groq API returned %d on attempt %d/%d, retrying in %.1fs...",
                    e.code, attempt + 1, max_retries, delay,
                )
                time.sleep(delay)
            else:
                raise

    raise RuntimeError(f"Groq API call failed after {max_retries} retries")
