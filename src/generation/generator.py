"""
Reply generation.

NOT EXECUTED AGAINST THE REAL API IN THIS SANDBOX: no network access, so no
real Claude calls were made during this development session. The
`AnthropicGenerator` class is real, correct client code (same auth
pattern the assignment specifies - API key from environment, never
hardcoded) and is meant to run on your machine with ANTHROPIC_API_KEY set.

`MockGenerator` is an explicit, clearly-labeled local fallback used only
so the rest of the pipeline (escalation, evidence formatting, output
schema) can be exercised without an API key. Its output must never be
reported as real evaluation results - see REPORT.md "What is misleading
about my headline number?" and scripts/evaluate.py, which refuses to
write headline metrics using MockGenerator output without an explicit
--allow-mock flag.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

from src.config import SETTINGS
from src.generation.prompts import REPLY_SYSTEM_PROMPT, REPLY_USER_TEMPLATE, format_evidence_block


@dataclass
class GeneratedReply:
    reply: str
    confidence: float
    evidence_case_ids: list[str]
    is_mock: bool = False


class GroqGenerator:
    """Free, high-speed generator using Groq's API (e.g. llama-3.3-70b-versatile)."""

    def __init__(self, model: str | None = None) -> None:
        self.api_key = os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY not set. Get a free API key at https://console.groq.com")
        self.model = model or os.environ.get("SUPPORT_AGENT_LLM_MODEL", "llama-3.3-70b-versatile")

    def generate(self, brand: str, message: str, context: str, intent: str, evidence: list[dict]) -> GeneratedReply:
        from src.utils import groq_api_call

        system = REPLY_SYSTEM_PROMPT.format(brand=brand)
        user = REPLY_USER_TEMPLATE.format(
            message=message, context=context or "(none)", intent=intent, brand=brand,
            evidence_block=format_evidence_block(evidence),
        )

        data = groq_api_call(
            api_key=self.api_key,
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=400,
            temperature=0.2,
        )

        text = data["choices"][0]["message"]["content"].strip()
        avg_sim = sum(e.get("similarity", 0) for e in evidence) / len(evidence) if evidence else 0.0
        return GeneratedReply(
            reply=text, confidence=round(avg_sim, 2),
            evidence_case_ids=[str(e.get("conversation_id", "")) for e in evidence], is_mock=False,
        )


class AnthropicGenerator:
    """Real generator. Requires `pip install anthropic` and ANTHROPIC_API_KEY."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or SETTINGS.llm_model
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Use MockGenerator for offline development, "
                "or set the key to use the real generator."
            )

    def generate(self, brand: str, message: str, context: str, intent: str, evidence: list[dict]) -> GeneratedReply:
        import anthropic  # local import: optional heavy dep, only needed for real calls

        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        system = REPLY_SYSTEM_PROMPT.format(brand=brand)
        user = REPLY_USER_TEMPLATE.format(
            message=message, context=context or "(none)", intent=intent, brand=brand,
            evidence_block=format_evidence_block(evidence),
        )
        resp = client.messages.create(
            model=self.model, max_tokens=400, system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(block.text for block in resp.content if getattr(block, "type", None) == "text").strip()
        avg_sim = sum(e.get("similarity", 0) for e in evidence) / len(evidence) if evidence else 0.0
        return GeneratedReply(
            reply=text, confidence=round(avg_sim, 2),
            evidence_case_ids=[str(e.get("conversation_id", "")) for e in evidence], is_mock=False,
        )


class MockGenerator:
    """
    Offline development fallback ONLY. Deterministically returns the
    top-retrieved historical response verbatim (i.e. does no actual
    synthesis) so the pipeline shape can be tested without an API key.
    Never used to produce reported headline numbers.
    """

    def generate(self, brand: str, message: str, context: str, intent: str, evidence: list[dict]) -> GeneratedReply:
        if not evidence:
            return GeneratedReply(
                reply="[MOCK] No historical evidence found; a real system would likely escalate this.",
                confidence=0.0, evidence_case_ids=[], is_mock=True,
            )
        top = evidence[0]
        return GeneratedReply(
            reply=f"[MOCK, not LLM-generated] {top.get('historical_response', '')}",
            confidence=round(top.get("similarity", 0.0), 2),
            evidence_case_ids=[e.get("conversation_id", "") for e in evidence],
            is_mock=True,
        )
