"""Local, dependency-light text embeddings (signed feature hashing).

No model download, no network, deterministic across processes (blake2b, not Python's salted hash()). Word
unigrams/bigrams carry meaning; character trigrams make it tolerant to plurals, typos and counts.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

import numpy as np

_TOKEN = re.compile(r"[\w']+", re.UNICODE)


class HashingEmbedder:
    def __init__(
        self, dim: int = 512, *, word_weight: float = 1.0, bigram_weight: float = 0.7, char_weight: float = 0.25
    ):
        if dim < 16:
            raise ValueError("embedding dimension must be >= 16")
        self.dim = dim
        self.word_weight = word_weight
        self.bigram_weight = bigram_weight
        self.char_weight = char_weight
        self._cache: dict[str, tuple[int, float]] = {}

    def _slot(self, feature: str) -> tuple[int, float]:
        hit = self._cache.get(feature)
        if hit is None:
            value = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "little")
            hit = (value % self.dim, 1.0 if (value >> 63) & 1 else -1.0)
            if len(self._cache) < 200_000:
                self._cache[feature] = hit
        return hit

    def features(self, text: str) -> Iterable[tuple[str, float]]:
        words = _TOKEN.findall(text.lower())
        for word in words:
            yield "w:" + word, self.word_weight
            padded = f"#{word}#"
            for i in range(len(padded) - 2):
                yield "c:" + padded[i : i + 3], self.char_weight
        for first, second in zip(words, words[1:], strict=False):
            yield f"b:{first} {second}", self.bigram_weight

    def embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for feature, weight in self.features(text):
            index, sign = self._slot(feature)
            vector[index] += sign * weight
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 0 else vector

    def to_bytes(self, vector: np.ndarray) -> bytes:
        return vector.astype(np.float16).tobytes()

    def from_bytes(self, blob: bytes) -> np.ndarray:
        return np.frombuffer(blob, dtype=np.float16).astype(np.float32)


def cosine_many(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity of one unit vector against rows of unit vectors."""
    if matrix.size == 0:
        return np.zeros(0, dtype=np.float32)
    return matrix @ query
