import numpy as np

from src.retrieval.embeddings import Embedder
from src.retrieval.index import VectorIndex


def test_embedder_encodes_texts():
    embedder = Embedder()
    texts = ["Order not received", "App crashes on login"]
    vecs = embedder.encode(texts)
    assert isinstance(vecs, np.ndarray)
    assert len(vecs) == 2
    assert vecs.shape[1] > 0


def test_vector_index_add_and_search():
    index = VectorIndex(dim=4)
    # Unit vectors
    v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    meta = [
        {"id": "doc1", "text": "shipping delay"},
        {"id": "doc2", "text": "password reset"},
    ]
    index.add(np.vstack([v1, v2]), meta)

    results = index.search(v1, top_k=2)
    assert len(results) == 2
    assert results[0]["id"] == "doc1"
    assert round(results[0]["similarity"], 2) == 1.0
    assert round(results[1]["similarity"], 2) == 0.0
