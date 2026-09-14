"""Sentence embeddings for the knowledge base.

Deliberately CPU-only. The 4 GB card is already holding llama3.2:3b (~2.3 GB)
and RoBERTa; a third resident model is what pushes generation into system RAM,
and Day 12 measured what that costs. Embedding a query is a few milliseconds on
CPU either way -- this is not the slow part of anything.

Model choice is a Day 14 measurement, not an assumption. all-MiniLM-L6-v2 is the
starting point because it is small (~90 MB), fast, and adequate on short
regulatory prose; bge-small-en-v1.5 is the upgrade to try if recall@5 comes back
poor. Change it with SYNTHEIA_EMBED_MODEL rather than editing code, so the
retrieval evaluation can sweep models.
"""

from __future__ import annotations

import os

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_model = None
_model_name: str | None = None


def get_model_name() -> str:
    return os.getenv("SYNTHEIA_EMBED_MODEL", DEFAULT_MODEL)


def get_model():
    """Load and cache the encoder. First call downloads ~90 MB."""
    global _model, _model_name
    name = get_model_name()
    if _model is not None and _model_name == name:
        return _model

    from sentence_transformers import SentenceTransformer

    _model = SentenceTransformer(name, device="cpu")
    _model_name = name
    return _model


def embed(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Encode documents. Normalised, so cosine similarity is a dot product."""
    if not texts:
        return []
    vecs = get_model().encode(
        texts, batch_size=batch_size, normalize_embeddings=True,
        show_progress_bar=False, convert_to_numpy=True,
    )
    return vecs.tolist()


def embed_query(text: str) -> list[float]:
    return embed([text])[0]


def dimension() -> int:
    return int(get_model().get_embedding_dimension())
