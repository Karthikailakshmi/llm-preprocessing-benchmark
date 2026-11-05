"""Chunk records, the baseline chunker and the vector index (same for both bots).

Embeddings come from fastembed (multilingual MiniLM by default, local, free).
If it cannot load (e.g. no internet for the first model download) the index
falls back to TF-IDF and reports that in `backend`, so the fallback is never
silent and is logged with every result.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Union

import numpy as np

from shared.config import EMBEDDING_MODEL


@dataclass
class Chunk:
    chunk_id: int
    text: str
    source: str
    page: Optional[Union[int, str]] = None
    unit: str = ""

    @property
    def ref(self) -> str:
        return f"{self.source}, {self.unit} {self.page}" if self.page is not None and self.unit \
            else self.source


def fixed_window_split(text: str, size: int = 800, overlap: int = 120) -> list[str]:
    """Baseline chunker: fixed character windows with overlap."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    step, out, i = max(size - overlap, 1), [], 0
    while True:
        piece = text[i:i + size].strip()
        if piece:
            out.append(piece)
        if i + size >= len(text):
            break
        i += step
    return out


def build_chunks(segments, size: int, overlap: int, splitter=fixed_window_split) -> list[Chunk]:
    chunks: list[Chunk] = []
    for seg in segments:
        for piece in splitter(seg.text, size, overlap):
            chunks.append(Chunk(len(chunks), piece, seg.source, seg.page, seg.unit))
    return chunks


@lru_cache(maxsize=1)
def _fastembed():
    try:
        from fastembed import TextEmbedding
        return TextEmbedding(model_name=EMBEDDING_MODEL)
    except Exception:  # noqa: BLE001
        return None


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.clip(norms, 1e-12, None)


class VectorIndex:
    def __init__(self):
        self.chunks: list[Chunk] = []
        self.backend = "empty"
        self._emb = None
        self._matrix = None
        self._tfidf = None

    def build(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("No chunks to index.")
        self.chunks = chunks
        texts = [c.text for c in chunks]
        self._emb = _fastembed()
        if self._emb is not None:
            vecs = np.array(list(self._emb.embed(texts)), dtype=np.float32)
            self._matrix = _normalize(vecs)
            self.backend = f"fastembed:{EMBEDDING_MODEL}"
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._tfidf = TfidfVectorizer(sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")
            self._matrix = self._tfidf.fit_transform(texts)
            self.backend = "tfidf-fallback"

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        if self._matrix is None:
            return []
        if self._emb is not None:
            q = _normalize(np.array(list(self._emb.embed([query])), dtype=np.float32))
            scores = (self._matrix @ q.T).ravel()
        else:
            scores = (self._matrix @ self._tfidf.transform([query]).T).toarray().ravel()
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]
