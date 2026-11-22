"""Semantic similarity between short texts, for the analysis agents.

Reuses the knowledge base's sentence encoder (`src/rag/embedder.py`): already
on disk, CPU-only by design so it never competes with the LLM for the 4 GB
card, and a few milliseconds per batch. No new model, no new dependency.

Similarity is a *candidate generator*, never a verdict. Two requirements about
the same subject score high whether they agree or contradict each other, and a
rule's template scores high against a requirement that only shares its
vocabulary. Every agent that uses this either confirms the candidate another
way or labels it unconfirmed.

Every caller takes a `Similarity` argument rather than importing the encoder, so
tests can pass a deterministic stand-in and stay offline.
"""

from __future__ import annotations

import re
from typing import Protocol

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {"the", "shall", "system", "must", "will", "should", "and", "for", "with",
         "that", "this", "from", "are", "all", "any", "each", "able", "be", "to",
         "of", "a", "an", "in", "on", "by", "or", "is", "as", "at", "its", "their"}


class Similarity(Protocol):
    def cross(self, a: list[str], b: list[str]) -> list[list[float]]:
        """Cosine similarity of every text in `a` against every text in `b`."""


class EmbeddingSimilarity:
    """Cosine similarity over the knowledge base encoder."""

    def cross(self, a: list[str], b: list[str]) -> list[list[float]]:
        if not a or not b:
            return [[] for _ in a]
        from src.rag.embedder import embed

        va, vb = embed(a), embed(b)
        # Vectors are normalised by `embed`, so the dot product is the cosine.
        return [[sum(x * y for x, y in zip(u, v)) for v in vb] for u in va]


class LexicalSimilarity:
    """Content-word Jaccard. Offline stand-in for tests and a degraded fallback.

    Much weaker than embeddings -- it cannot see that "retain" and "preserve"
    mean the same thing -- so it is used only when the encoder cannot load.
    """

    @staticmethod
    def _words(text: str) -> set[str]:
        return {w for w in _TOKEN.findall(text.lower()) if w not in _STOP and len(w) > 2}

    def cross(self, a: list[str], b: list[str]) -> list[list[float]]:
        wb = [self._words(t) for t in b]
        out = []
        for t in a:
            wa = self._words(t)
            out.append([len(wa & w) / len(wa | w) if wa | w else 0.0 for w in wb])
        return out


def default_similarity() -> Similarity:
    """The encoder if it loads, else the lexical fallback -- never an exception.

    A run whose embedding model fails to load should still produce analysis,
    just less of it, rather than losing everything after extraction.
    """
    try:
        from src.rag.embedder import get_model

        get_model()
        return EmbeddingSimilarity()
    except Exception as e:                                   # noqa: BLE001
        print(f"  (similarity: encoder unavailable, using lexical fallback -- "
              f"{type(e).__name__})")
        return LexicalSimilarity()


def content_words(text: str) -> set[str]:
    return LexicalSimilarity._words(text)
