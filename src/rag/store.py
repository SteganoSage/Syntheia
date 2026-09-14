"""ChromaDB persistence for knowledge-base chunks.

Embeddings are computed by `src.rag.embedder` and handed to Chroma explicitly
rather than letting Chroma pick its own embedding function. Two reasons: the
model then stays a single configurable choice that the Day 14 retrieval
evaluation can sweep, and queries and documents are guaranteed to be encoded by
the same model -- a mismatch there produces silently terrible retrieval with no
error, which is precisely the class of bug this project keeps meeting.
"""

from __future__ import annotations

from pathlib import Path

from src.rag import embedder
from src.rag.schema import Chunk, SourceDoc

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DIR = REPO / "data/vector_store"
COLLECTION = "knowledge_base"


def get_client(persist_dir: Path | None = None):
    import chromadb

    target = Path(persist_dir or DEFAULT_DIR)
    target.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(target))


def get_collection(persist_dir: Path | None = None, reset: bool = False):
    client = get_client(persist_dir)
    if reset:
        try:
            client.delete_collection(COLLECTION)
        except Exception:                       # noqa: BLE001 -- absent is fine
            pass
    # Cosine, to match the normalised vectors the embedder produces.
    return client.get_or_create_collection(
        COLLECTION, metadata={"hnsw:space": "cosine"}
    )


def add_chunks(chunks: list[Chunk], persist_dir: Path | None = None,
               collection=None, batch_size: int = 128) -> int:
    """Embed and store. Returns how many were written."""
    if not chunks:
        return 0
    col = collection or get_collection(persist_dir)
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        col.upsert(
            ids=[c.chunk_id for c in batch],
            documents=[c.text for c in batch],
            embeddings=embedder.embed([c.text for c in batch]),
            metadatas=[c.metadata() for c in batch],
        )
    return len(chunks)


def query(text: str, k: int = 5, where: dict | None = None,
          persist_dir: Path | None = None, collection=None) -> list[dict]:
    """Top-k chunks for a query, nearest first.

    `score` is cosine similarity in [0, 1] -- Chroma returns cosine *distance*,
    so it is inverted here. That score becomes the confidence attached to a
    citation, and the threshold below which §6 says to escalate for human
    review, so it must mean "more is better" at the call site.
    """
    col = collection or get_collection(persist_dir)
    if col.count() == 0:
        return []
    res = col.query(
        query_embeddings=[embedder.embed_query(text)],
        n_results=min(k, col.count()),
        where=where or None,
        include=["documents", "metadatas", "distances"],
    )
    out = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        out.append({"text": doc, "metadata": meta, "score": round(1.0 - float(dist), 4)})
    return out


def stats(persist_dir: Path | None = None) -> dict:
    col = get_collection(persist_dir)
    return {"chunks": col.count(), "embedding_model": embedder.get_model_name()}
