"""Built-in dependency-free embedder.

A deterministic feature-hashing character n-gram embedder so the memory layer
works offline out of the box (tests, demos, air-gapped agents).  Swap in any
callable `text -> np.ndarray` — e.g. a sentence-transformers model or an
embeddings API — for production-quality semantics.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np


class HashingEmbedder:
    """L2-normalized char-n-gram feature hashing into a fixed dimension."""

    def __init__(self, dim: int = 256, ngram_range: tuple[int, int] = (3, 5)):
        self.dim = dim
        self.ngram_range = ngram_range

    def __call__(self, text: str) -> np.ndarray:
        normalized = " " + re.sub(r"\s+", " ", text.lower().strip()) + " "
        vec = np.zeros(self.dim, dtype=np.float64)
        lo, hi = self.ngram_range
        for n in range(lo, hi + 1):
            for i in range(max(0, len(normalized) - n + 1)):
                gram = normalized[i:i + n]
                digest = hashlib.blake2b(gram.encode(), digest_size=8).digest()
                h = int.from_bytes(digest, "little")
                idx = h % self.dim
                sign = 1.0 if (h >> 63) & 1 else -1.0
                vec[idx] += sign
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec
