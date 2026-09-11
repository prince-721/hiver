"""
Sentence-transformer embeddings for the retrieval knowledge base.

NOT EXECUTED IN THIS DEVELOPMENT SANDBOX: sentence-transformers requires
downloading a pretrained model from the internet, and this environment has
no network access. This module is written to the real interface and is
meant to run on your machine (`pip install -r requirements.txt` will pull
it down normally). See README "Known unexecuted paths".
"""
from __future__ import annotations

import hashlib
import re

import numpy as np

from src.config import SETTINGS


def _hash_vector(text: str, dim: int = 128) -> np.ndarray:
    """Deterministic hash-based bag-of-ngrams embedding fallback when offline."""
    vec = np.zeros(dim, dtype=np.float32)
    tokens = re.findall(r"\b\w+\b", text.lower())
    for token in tokens:
        idx = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % dim
        vec[idx] += 1.0
    for i in range(len(tokens) - 1):
        bigram = f"{tokens[i]}_{tokens[i+1]}"
        idx = int(hashlib.md5(bigram.encode("utf-8")).hexdigest(), 16) % dim
        vec[idx] += 1.5
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


class Embedder:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or SETTINGS.embedding_model
        self._model = None
        self._use_fallback = False

    def _load(self):
        if self._model is None and not self._use_fallback:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except Exception:
                self._use_fallback = True
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        if model is not None and not self._use_fallback:
            return np.asarray(model.encode(texts, normalize_embeddings=True, show_progress_bar=False))
        # Fallback offline embedding
        dim = 128
        matrix = np.zeros((len(texts), dim), dtype=np.float32)
        for i, txt in enumerate(texts):
            matrix[i] = _hash_vector(txt, dim)
        return matrix
