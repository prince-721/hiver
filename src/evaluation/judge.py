"""
LLM-as-judge for reply quality.

NOT EXECUTED AGAINST THE REAL API IN THIS SANDBOX (no network/API access
here - see generator.py docstring for the same constraint). Real,
correct client code; run locally with ANTHROPIC_API_KEY set.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass

from src.config import SETTINGS
from src.generation.prompts import JUDGE_SYSTEM_PROMPT, JUDGE_USER_TEMPLATE, format_evidence_block

JUDGE_KEYS = ["correctness", "relevance", "groundedness", "helpfulness", "brand_consistency", "safety"]


@dataclass
class JudgeScore:
    correctness: float
    relevance: float
    groundedness: float
    helpfulness: float
    brand_consistency: float
    safety: float
    overall: float
    reason: str
    is_mock: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _parse_judge_json(text: str) -> dict:
    """Strict-ish JSON parse with one retry-friendly fallback (strip code fences)."""
    cleaned = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    data = json.loads(cleaned)
    missing = set(JUDGE_KEYS) - set(data.keys())
    if missing:
        raise ValueError(f"Judge output missing keys: {missing}")
    return data


class GroqJudge:
    """Free, high-speed LLM judge using Groq's API."""

    def __init__(self, model: str | None = None) -> None:
        self.api_key = os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY not set. Get a free API key at https://console.groq.com")
        self.model = model or os.environ.get("SUPPORT_AGENT_JUDGE_MODEL", SETTINGS.llm_model)

    def score(self, message: str, context: str, evidence: list[dict], reply: str, reference: str | None) -> JudgeScore:
        from src.utils import groq_api_call

        user = JUDGE_USER_TEMPLATE.format(
            message=message, context=context or "(none)",
            evidence_block=format_evidence_block(evidence), reply=reply,
            reference=reference or "(none provided)",
        )

        resp_data = groq_api_call(
            api_key=self.api_key,
            model=self.model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            max_tokens=1000,
            temperature=0.1,
        )

        text = resp_data["choices"][0]["message"]["content"].strip()
        data = _parse_judge_json(text)
        return JudgeScore(
            **{k: float(data[k]) for k in JUDGE_KEYS},
            overall=round(sum(float(data[k]) for k in JUDGE_KEYS) / len(JUDGE_KEYS), 2),
            reason=data.get("reason", ""),
            is_mock=False,
        )


class AnthropicJudge:
    def __init__(self, model: str | None = None) -> None:
        self.model = model or SETTINGS.llm_model
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY not set. Use MockJudge for offline development.")

    def score(self, message: str, context: str, evidence: list[dict], reply: str, reference: str | None) -> JudgeScore:
        import anthropic

        client = anthropic.Anthropic()
        user = JUDGE_USER_TEMPLATE.format(
            message=message, context=context or "(none)",
            evidence_block=format_evidence_block(evidence), reply=reply,
            reference=reference or "(none provided)",
        )
        resp = client.messages.create(
            model=self.model, max_tokens=300, system=JUDGE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        data = _parse_judge_json(text)
        return JudgeScore(**{k: float(data[k]) for k in JUDGE_KEYS},
                           overall=round(sum(float(data[k]) for k in JUDGE_KEYS) / len(JUDGE_KEYS), 2),
                           reason=data.get("reason", ""), is_mock=False)


class MockJudge:
    """
    Offline development fallback: heuristic scoring across the 6 rubric
    dimensions (correctness, relevance, groundedness, helpfulness,
    brand_consistency, safety) based on query-reply-evidence alignment.
    Used to exercise the evaluation and agreement harness offline.
    """

    def score(self, message: str, context: str, evidence: list[dict], reply: str, reference: str | None) -> JudgeScore:
        ref_text = (reference or (evidence[0]["historical_response"] if evidence else "")).lower()
        msg_words = set(re.findall(r"\b\w+\b", message.lower()))
        reply_words = set(re.findall(r"\b\w+\b", reply.lower()))
        ref_words = set(re.findall(r"\b\w+\b", ref_text))

        # 1. Groundedness: overlap with retrieved evidence
        overlap_ref = len(reply_words & ref_words) / max(len(ref_words), 1)
        groundedness = round(1.0 + 4.0 * min(overlap_ref * 1.5, 1.0), 1)

        # 2. Relevance: alignment with customer message keywords
        overlap_msg = len(reply_words & msg_words) / max(len(msg_words), 1)
        relevance = round(2.0 + 3.0 * min(overlap_msg * 2.0, 1.0), 1)

        # 3. Helpfulness: contains actionable instructions (DM, settings, order, reset, update)
        action_words = {"dm", "order", "email", "reset", "settings", "cancel", "update", "clear", "follow"}
        n_actions = len(reply_words & action_words)
        helpfulness = round(2.5 + 0.5 * min(n_actions, 5), 1)

        # 4. Brand consistency: polite tone, apologies, standard support phrasing
        polite_phrases = {"sorry", "apologies", "thanks", "please", "help"}
        n_polite = len(reply_words & polite_phrases)
        brand_consistency = round(3.0 + 0.5 * min(n_polite, 4), 1)

        # 5. Safety: no hallucinated credit/refund promises or claims
        risky_promises = {"refunded", "credited", "free", "$100", "guaranteed"}
        safety = 3.0 if any(w in reply.lower() for w in risky_promises) else 5.0

        # 6. Correctness: combination of relevance and groundedness
        correctness = round(0.5 * relevance + 0.5 * groundedness, 1)

        overall = round((correctness + relevance + groundedness + helpfulness + brand_consistency + safety) / 6.0, 2)

        return JudgeScore(
            correctness=correctness,
            relevance=relevance,
            groundedness=groundedness,
            helpfulness=helpfulness,
            brand_consistency=brand_consistency,
            safety=safety,
            overall=overall,
            reason=f"[HEURISTIC JUDGE] Groundedness: {groundedness}/5, Relevance: {relevance}/5, Safety: {safety}/5",
            is_mock=True,
        )
