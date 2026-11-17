"""Investigation-only HNSW retrieval using the supplied corpus vectors."""
from __future__ import annotations

import time

import numpy as np

from src.retriever import ExactDenseRetriever


class HNSWDenseRetriever(ExactDenseRetriever):
    """Approximate cosine search, reusing baseline vector loading/validation.

    Fixed insertion order, seed and one build/search thread improve repeatability.
    ef_search must cover k: changing it changes the search/quality trade-off.
    """

    def __init__(self, embeddings_path, ids_path, *, m=16,
                 ef_construction=200, ef_search=100, seed=42):
        if (type(m) is not int or m < 2
                or type(ef_construction) is not int or ef_construction < m
                or type(ef_search) is not int or ef_search < 1):
            raise ValueError("require m >= 2, ef_construction >= m and ef_search >= 1")
        try:
            import hnswlib
        except ImportError as exc:
            raise ImportError(
                "HNSW requires: pip install -r rag/extensions/requirements.txt"
            ) from exc
        super().__init__(embeddings_path, ids_path)
        self.ef_search = ef_search
        self.index = hnswlib.Index(space="ip", dim=self.embeddings.shape[1])
        self.index.init_index(max_elements=len(self.record_ids), M=m,
                              ef_construction=ef_construction, random_seed=seed)
        self.index.add_items(self.embeddings, np.arange(len(self.record_ids)), num_threads=1)
        self.index.set_num_threads(1)
        self.index.set_ef(ef_search)

    def retrieve(self, query_vector: np.ndarray, k: int = 5):
        if type(k) is not int or k < 1:
            raise ValueError("k must be a positive integer")
        q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if q.size != self.embeddings.shape[1] or not np.isfinite(q).all():
            raise ValueError("query vector must be finite and match embedding dimension")
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            raise ValueError("zero-norm query vector")
        q = q / q_norm
        k = min(k, len(self.record_ids))
        if k > self.ef_search:
            raise ValueError("ef_search must be at least the requested retrieval depth")
        start = time.perf_counter()
        labels, distances = self.index.knn_query(q.reshape(1, -1), k=k, num_threads=1)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return {
            "record_ids": [self.record_ids[int(i)] for i in labels[0]],
            "scores": [float(1.0 - distance) for distance in distances[0]],
            "latency_ms": elapsed_ms,
        }

