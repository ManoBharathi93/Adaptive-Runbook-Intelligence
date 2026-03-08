"""
Embedding utilities – wraps sentence-transformers for local embeddings.
Falls back gracefully if GPU is unavailable.
"""

from __future__ import annotations

import functools
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from utils.config import get_config

_cfg = get_config().retrieval


@functools.lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Load embedding model once and cache."""
    return SentenceTransformer(_cfg.embedding_model)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Return list of embedding vectors for a list of texts."""
    model = _get_model()
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return embeddings.tolist()


def embed_single(text: str) -> List[float]:
    """Embed a single string."""
    return embed_texts([text])[0]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two normalised vectors."""
    va = np.array(a)
    vb = np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)
