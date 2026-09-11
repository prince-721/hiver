"""
FastAPI wrapper around the same pipeline used by the CLI.

NOT EXECUTED IN THIS SANDBOX: fastapi/uvicorn aren't installed here (no
network to install them). Logic is identical to src/predict.py, which
IS exercised above via the CLI - only the HTTP layer is unverified.
Run locally: `uvicorn src.api.main:app --reload` then POST to /predict.
"""
from __future__ import annotations

from pydantic import BaseModel

from src.predict import build_pipeline

try:
    from fastapi import FastAPI
except ImportError as e:  # pragma: no cover
    raise ImportError("fastapi not installed - `pip install fastapi uvicorn` to run the API locally.") from e

app = FastAPI(title="Hiver Support Agent API")
_pipeline = None


class PredictRequest(BaseModel):
    message: str
    conversation_context: list[str] | None = None


@app.on_event("startup")
def _load_pipeline() -> None:
    global _pipeline
    _pipeline = build_pipeline(use_full=False)


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    context = " ".join(req.conversation_context) if req.conversation_context else ""
    result = _pipeline.predict(req.message, context)
    return {
        "intent": result.intent,
        "intent_confidence": result.intent_confidence,
        "reply": result.reply,
        "escalation": {
            "decision": result.escalation_decision,
            "reason": result.escalation_reason,
            "confidence": result.escalation_confidence,
        },
        "evidence": result.evidence,
        "mode": result.mode,
    }
