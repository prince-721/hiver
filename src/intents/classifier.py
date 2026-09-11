"""
Intent Classifier module per assignment STEP 8 and STEP 23.

Implements an explainable semantic classifier:
  1. Embeds text using the Embedder (or TF-IDF fallback if offline).
  2. Fits class centroids (mean normalized embedding per intent class).
  3. Predicts intent by cosine similarity to class centroids.
  4. Computes calibrated confidence via temperature-scaled softmax.
  5. Returns structured JSON:
       {
         "intent": str,
         "confidence": float,
         "top_candidates": list[str]
       }
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.retrieval.embeddings import Embedder


class IntentClassifier:
    def __init__(self, embedder: Embedder | None = None, temperature: float = 0.2) -> None:
        self.embedder = embedder or Embedder()
        self.temperature = temperature
        self.centroids: dict[str, np.ndarray] = {}
        self.classes_: list[str] = []

    def fit(self, messages: list[str] | pd.Series, labels: list[str] | pd.Series) -> "IntentClassifier":
        msg_list = list(messages)
        lbl_list = list(labels)
        assert len(msg_list) == len(lbl_list), "messages and labels must have equal length"

        embeddings = self.embedder.encode(msg_list)
        unique_labels = sorted(set(lbl_list))
        self.classes_ = unique_labels
        self.centroids = {}

        for lb in unique_labels:
            indices = [i for i, l in enumerate(lbl_list) if l == lb]
            class_vecs = embeddings[indices]
            centroid = np.mean(class_vecs, axis=0)
            norm = np.linalg.norm(centroid)
            self.centroids[lb] = centroid / norm if norm > 0 else centroid

        return self

    def predict(self, message: str) -> dict:
        if not self.centroids:
            raise RuntimeError("Classifier has not been fitted. Call fit() first.")

        query_vec = self.embedder.encode([message])[0]
        q_norm = np.linalg.norm(query_vec)
        if q_norm > 0:
            query_vec = query_vec / q_norm

        scores = []
        for c in self.classes_:
            sim = float(np.dot(query_vec, self.centroids[c]))
            scores.append(sim)

        scores_arr = np.array(scores)
        scaled = scores_arr / self.temperature
        exp_vals = np.exp(scaled - np.max(scaled))
        probs = exp_vals / np.sum(exp_vals)

        order = np.argsort(probs)[::-1]
        top_idx = order[0]

        return {
            "intent": self.classes_[top_idx],
            "confidence": round(float(probs[top_idx]), 4),
            "top_candidates": [self.classes_[i] for i in order[:3]],
        }
