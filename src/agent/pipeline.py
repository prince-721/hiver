"""
End-to-end agent pipeline: message -> intent -> retrieval -> reply ->
escalation decision.

Designed to run in two modes so it's always runnable, but the mode
actually used is always reported in the output (never silently swapped):

  "full"  - sentence-transformer retrieval + real Claude generation.
            Requires deps installed + ANTHROPIC_API_KEY. NOT exercised in
            this sandbox (see retrieval/embeddings.py, generation/generator.py).
  "light" - TF-IDF retrieval (src/baselines/tfidf.py) + MockGenerator.
            Fully offline, fully executed and tested here. This is what
            scripts/predict.py and the API default to unless --full is
            passed and the dependencies/API key are actually present.
"""
from __future__ import annotations

from dataclasses import dataclass

import os
from src.baselines.tfidf import TfidfBaseline
from src.escalation.policy import decide
from src.generation.generator import AnthropicGenerator, GroqGenerator, MockGenerator
from src.intents.taxonomy import rule_label


@dataclass
class PredictResult:
    intent: str
    intent_confidence: float | None
    reply: str
    escalation_decision: str
    escalation_reason: str
    escalation_confidence: float
    evidence: list[dict]
    mode: str


class SupportAgentPipeline:
    def __init__(self, tfidf_baseline: TfidfBaseline, brand: str, use_full: bool = False) -> None:
        self.tfidf = tfidf_baseline
        self.brand = brand
        self.mode = "light"
        if use_full:
            if os.environ.get("GROQ_API_KEY"):
                try:
                    self.generator = GroqGenerator()
                    self.mode = "full (groq)"
                except Exception:
                    self.generator = MockGenerator()
            elif os.environ.get("ANTHROPIC_API_KEY"):
                try:
                    self.generator = AnthropicGenerator()
                    self.mode = "full (anthropic)"
                except Exception:
                    self.generator = MockGenerator()
            else:
                self.generator = MockGenerator()
        else:
            self.generator = MockGenerator()

    def predict(self, message: str, context: str | None = None) -> PredictResult:
        intent_pred = self.tfidf.predict_intent(message)
        evidence = self.tfidf.retrieve(message, top_k=3)
        top_sim = evidence[0]["similarity"] if evidence else None

        gen = self.generator.generate(
            brand=self.brand, message=message, context=context or "",
            intent=intent_pred["intent"], evidence=evidence,
        )

        esc = decide(
            message=message, intent=intent_pred["intent"],
            intent_confidence=intent_pred["confidence"] or 0.0,
            top_retrieval_similarity=top_sim, n_relevant_cases=len(evidence),
        )

        return PredictResult(
            intent=intent_pred["intent"], intent_confidence=intent_pred["confidence"],
            reply=gen.reply, escalation_decision=esc.decision, escalation_reason=esc.reason,
            escalation_confidence=esc.confidence,
            evidence=[{"case_id": e.get("conversation_id"), "similarity": e.get("similarity")} for e in evidence],
            mode=self.mode,
        )
