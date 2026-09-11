"""
Baseline 2 (simple, non-LLM): TF-IDF + Logistic Regression for intent,
TF-IDF cosine-similarity retrieval for the reply (return the historical
response of the most similar training message), and a similarity
threshold for escalation.

This is the meaningful comparison point for the full AI system - if the
LLM+embeddings system can't beat this, that is a real finding, not a
bug to hide.
"""
from __future__ import annotations

import math
import re
from collections import Counter
import numpy as np
import pandas as pd

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics.pairwise import cosine_similarity
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

from src.config import SETTINGS

_STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "cannot", "could", "did", "do",
    "does", "doing", "don't", "down", "during", "each", "few", "for", "from", "further",
    "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself",
    "me", "more", "most", "my", "myself", "no", "nor", "not", "of", "off", "on",
    "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over",
    "own", "same", "she", "should", "so", "some", "such", "than", "that", "the",
    "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "we",
    "were", "what", "when", "where", "which", "while", "who", "whom", "why", "with",
    "you", "your", "yours", "yourself", "yourselves"
}


class _SimpleTfidfVectorizer:
    """Lightweight pure-NumPy TF-IDF vectorizer when sklearn is not installed."""
    def __init__(self, max_features: int = 5000) -> None:
        self.max_features = max_features
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None

    def _tokenize(self, text: str) -> list[str]:
        words = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())
        tokens = [w for w in words if w not in _STOP_WORDS]
        bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]
        return tokens + bigrams

    def fit_transform(self, raw_documents: list[str]) -> np.ndarray:
        doc_tokens = [self._tokenize(doc) for doc in raw_documents]
        n_docs = len(raw_documents)
        df_counts = Counter()
        for toks in doc_tokens:
            df_counts.update(set(toks))

        most_common = df_counts.most_common(self.max_features)
        self.vocab = {term: i for i, (term, _) in enumerate(most_common)}
        
        idf_vals = np.zeros(len(self.vocab), dtype=np.float32)
        for term, idx in self.vocab.items():
            idf_vals[idx] = math.log((1 + n_docs) / (1 + df_counts[term])) + 1.0
        self.idf = idf_vals

        return self._transform_tokens(doc_tokens)

    def _transform_tokens(self, doc_tokens: list[list[str]]) -> np.ndarray:
        n_docs = len(doc_tokens)
        n_terms = len(self.vocab)
        matrix = np.zeros((n_docs, n_terms), dtype=np.float32)
        for d_idx, toks in enumerate(doc_tokens):
            counts = Counter(toks)
            for term, cnt in counts.items():
                if term in self.vocab:
                    t_idx = self.vocab[term]
                    matrix[d_idx, t_idx] = cnt * self.idf[t_idx]
            norm = np.linalg.norm(matrix[d_idx])
            if norm > 0:
                matrix[d_idx] /= norm
        return matrix

    def transform(self, raw_documents: list[str]) -> np.ndarray:
        doc_tokens = [self._tokenize(doc) for doc in raw_documents]
        return self._transform_tokens(doc_tokens)


class TfidfBaseline:
    def __init__(self, similarity_escalation_threshold: float | None = None) -> None:
        if _SKLEARN_AVAILABLE:
            self.vec = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
            self.clf = LogisticRegression(max_iter=1000)
        else:
            self.vec = _SimpleTfidfVectorizer(max_features=5000)
            self.clf = None
        self.threshold = similarity_escalation_threshold or SETTINGS.min_retrieval_similarity
        self._train_messages: list[str] = []
        self._train_responses: list[str] = []
        self._train_ids: list = []
        self._train_matrix = None
        self._centroids: dict[str, np.ndarray] = {}

    def fit(self, messages: pd.Series, intents: pd.Series, responses: pd.Series, ids: pd.Series | None = None) -> "TfidfBaseline":
        msg_list = list(messages)
        intent_list = list(intents)
        X = self.vec.fit_transform(msg_list)
        if _SKLEARN_AVAILABLE:
            self.clf.fit(X, intent_list)
        else:
            self._centroids = {}
            for label in set(intent_list):
                indices = [i for i, lb in enumerate(intent_list) if lb == label]
                sub_mat = X[indices]
                mean_vec = np.mean(sub_mat, axis=0)
                norm = np.linalg.norm(mean_vec)
                self._centroids[label] = mean_vec / norm if norm > 0 else mean_vec

        self._train_messages = msg_list
        self._train_responses = list(responses)
        self._train_ids = list(ids) if ids is not None else list(range(len(msg_list)))
        self._train_matrix = X
        return self

    def predict_intent(self, message: str) -> dict:
        x = self.vec.transform([message])
        if _SKLEARN_AVAILABLE:
            proba = self.clf.predict_proba(x)[0]
            classes = self.clf.classes_
            order = np.argsort(proba)[::-1]
            top_idx = order[0]
            return {
                "intent": classes[top_idx],
                "confidence": round(float(proba[top_idx]), 4),
                "top_candidates": [classes[i] for i in order[:3]],
            }
        else:
            classes = list(self._centroids.keys())
            sims = np.array([float(np.dot(x[0], self._centroids[c])) for c in classes])
            # Softmax with temperature 0.2 for calibrated confidence
            scaled = sims / 0.2
            exp_s = np.exp(scaled - np.max(scaled))
            proba = exp_s / np.sum(exp_s)
            order = np.argsort(proba)[::-1]
            top_idx = order[0]
            return {
                "intent": classes[top_idx],
                "confidence": round(float(proba[top_idx]), 4),
                "top_candidates": [classes[i] for i in order[:3]],
            }

    def retrieve(self, message: str, top_k: int = 3) -> list[dict]:
        x = self.vec.transform([message])
        if _SKLEARN_AVAILABLE:
            sims = cosine_similarity(x, self._train_matrix)[0]
        else:
            sims = np.dot(self._train_matrix, x[0])
        order = np.argsort(sims)[::-1][:top_k]
        return [
            {
                "similarity": round(float(sims[i]), 4),
                "historical_message": self._train_messages[i],
                "historical_response": self._train_responses[i],
                "conversation_id": self._train_ids[i],
            }
            for i in order
        ]

    def predict_reply(self, message: str) -> tuple[str, list[dict]]:
        hits = self.retrieve(message, top_k=1)
        if not hits:
            return "", []
        return hits[0]["historical_response"], hits

    def predict_escalation(self, top_similarity: float) -> dict:
        if top_similarity < self.threshold:
            return {
                "decision": "ESCALATE",
                "reason": f"Top TF-IDF similarity {top_similarity:.2f} below threshold {self.threshold}.",
                "confidence": 1 - top_similarity,
            }
        return {
            "decision": "AUTO_HANDLE",
            "reason": f"Top TF-IDF similarity {top_similarity:.2f} meets threshold.",
            "confidence": top_similarity,
        }
