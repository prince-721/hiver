"""
Thin FAISS wrapper for the historical-resolution knowledge base.

NOT EXECUTED IN THIS DEVELOPMENT SANDBOX (no network to install faiss-cpu).
Written to the real interface; run locally after `pip install -r requirements.txt`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class VectorIndex:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.metadata: list[dict] = []
        self._vectors: np.ndarray | None = None
        self._faiss_index = None
        try:
            import faiss
            self._faiss_index = faiss.IndexFlatIP(dim)
        except ImportError:
            self._faiss_index = None

    def add(self, vectors: np.ndarray, metadata: list[dict]) -> None:
        assert len(vectors) == len(metadata)
        vecs = vectors.astype("float32")
        if self._faiss_index is not None:
            self._faiss_index.add(vecs)
        else:
            if self._vectors is None:
                self._vectors = vecs
            else:
                self._vectors = np.vstack([self._vectors, vecs])
        self.metadata.extend(metadata)

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> list[dict]:
        q = query_vec.astype("float32").reshape(1, -1)
        if self._faiss_index is not None:
            scores, idxs = self._faiss_index.search(q, top_k)
            results = []
            for score, idx in zip(scores[0], idxs[0]):
                if idx == -1:
                    continue
                results.append({**self.metadata[idx], "similarity": round(float(score), 4)})
            return results
        else:
            if self._vectors is None or len(self._vectors) == 0:
                return []
            scores = np.dot(self._vectors, q.flatten())
            k = min(top_k, len(scores))
            order = np.argsort(scores)[::-1][:k]
            results = []
            for idx in order:
                results.append({**self.metadata[idx], "similarity": round(float(scores[idx]), 4)})
            return results

    def save(self, path: str) -> None:
        if self._faiss_index is not None:
            import faiss
            faiss.write_index(self._faiss_index, f"{path}.faiss")
        elif self._vectors is not None:
            np.save(f"{path}.npy", self._vectors)
        Path(f"{path}.meta.json").write_text(json.dumps(self.metadata, indent=2))

    @classmethod
    def load(cls, path: str, dim: int) -> "VectorIndex":
        obj = cls(dim)
        meta_file = Path(f"{path}.meta.json")
        if meta_file.exists():
            obj.metadata = json.loads(meta_file.read_text())

        faiss_file = Path(f"{path}.faiss")
        npy_file = Path(f"{path}.npy")
        if faiss_file.exists() and obj._faiss_index is not None:
            import faiss
            obj._faiss_index = faiss.read_index(str(faiss_file))
        elif npy_file.exists():
            obj._vectors = np.load(str(npy_file))
        return obj
