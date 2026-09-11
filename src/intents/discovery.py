"""
Discovers candidate intent clusters from the brand's own customer
messages using TF-IDF + KMeans (deliberately not embeddings here - this
step is meant to be cheap, fast, and inspectable by a human before
intents are frozen into data/intent_taxonomy.json).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

from src.baselines.tfidf import _SimpleTfidfVectorizer


def _simple_kmeans(X: np.ndarray, k: int, random_state: int = 42, max_iter: int = 20) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    n_samples = X.shape[0]
    init_indices = rng.choice(n_samples, size=k, replace=False)
    centroids = X[init_indices].copy()
    labels = np.zeros(n_samples, dtype=int)

    for _ in range(max_iter):
        sims = np.dot(X, centroids.T)
        new_labels = np.argmax(sims, axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for c in range(k):
            members = X[labels == c]
            if len(members) > 0:
                mean_c = np.mean(members, axis=0)
                norm = np.linalg.norm(mean_c)
                centroids[c] = mean_c / norm if norm > 0 else mean_c
    return labels, centroids


def discover_clusters(
    messages: pd.Series, n_clusters: int = 8, random_state: int = 42, top_terms: int = 8
) -> tuple[pd.Series, dict[int, list[str]]]:
    """
    Returns (cluster_assignment per message, {cluster_id: top_terms}) so a
    human can name each cluster into a final intent label.
    """
    k = min(n_clusters, max(2, len(messages) // 10))  # don't over-cluster tiny samples

    if _SKLEARN_AVAILABLE:
        vec = TfidfVectorizer(max_features=2000, stop_words="english", ngram_range=(1, 2))
        X = vec.fit_transform(messages)
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        terms = vec.get_feature_names_out()
        top_terms_per_cluster = {}
        for c in range(k):
            center = km.cluster_centers_[c]
            top_idx = center.argsort()[::-1][:top_terms]
            top_terms_per_cluster[c] = [terms[i] for i in top_idx]
        return pd.Series(labels, index=messages.index, name="cluster"), top_terms_per_cluster
    else:
        vec = _SimpleTfidfVectorizer(max_features=2000)
        X = vec.fit_transform(list(messages))
        labels, centroids = _simple_kmeans(X, k=k, random_state=random_state)
        # Invert vocab
        idx_to_term = {idx: term for term, idx in vec.vocab.items()}
        top_terms_per_cluster = {}
        for c in range(k):
            center = centroids[c]
            top_idx = center.argsort()[::-1][:top_terms]
            top_terms_per_cluster[c] = [idx_to_term[i] for i in top_idx if i in idx_to_term]
        return pd.Series(labels, index=messages.index, name="cluster"), top_terms_per_cluster
