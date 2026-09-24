"""Documents handed to a single run, indexed beside the curated corpus.

The supervisor's part 1 is "feed a lightweight LLM existing requirement
documents". The curated knowledge base is one way to do that, ingested ahead of
time with a manifest per document. This is the other: hand the system a PDF
along with the scenario and have it cite from that too.

It is not a convenience. `data/knowledge_base/sources.yaml` carries seven
`unavailable:` entries -- documents that were sought and could not be got --
and IRDAI is one of them. Every insurance run therefore prints "no regulation in
the knowledge base supports this" against its IRDAI requirements, and no amount
of retrieval tuning fixes that, because the corpus simply does not contain the
document. Supplying it at run time is the only thing that can.

Three rules, all of them about not letting a supplied document impersonate a
curated one:

**A separate store.** The session gets its own ChromaDB directory, which is
deleted when the run ends. Adding to the curated collection would mean one demo
permanently injecting unmanifested documents into the thing whose entire selling
point is versioned provenance.

**No boost.** Supplied chunks compete on the same cosine score against the same
`MIN_CONFIDENCE` floor. Ranking them higher because the user chose them would be
an unearned claim, and would let a loosely-related PDF outrank a real clause.

**Marked citations.** Every citation from here carries `[supplied]`, because the
issuer, jurisdiction and effective date §5 asks for are genuinely unknown.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from src.rag import chunker, loader, store
from src.rag.schema import Chunk


class SuppliedDocError(RuntimeError):
    """A supplied document could not be used. Always reported, never silent."""


def _collect(paths: list[Path]) -> list[Path]:
    """Expand directories, keep only what the loader can read."""
    out: list[Path] = []
    for p in paths:
        if p.is_dir():
            out.extend(sorted(loader.discover(p)))
        elif p.is_file():
            out.append(p)
        else:
            raise SuppliedDocError(f"no such file or directory: {p}")
    return out


class SessionCorpus:
    """Supplied documents, indexed for one run and then thrown away.

    Use as a context manager so the temporary store is removed even when the
    run fails:

        with SessionCorpus([Path("irdai.pdf")]) as sess:
            hits = store.query(text, persist_dir=sess.persist_dir)
    """

    def __init__(self, paths: list[Path]):
        self.paths = _collect(paths)
        self.persist_dir: Path | None = None
        self.chunks: list[Chunk] = []
        # Per document: what happened. Reported rather than summed, because
        # "3 of 4 indexed" is useless without knowing which one failed and why.
        self.report: list[tuple[str, str]] = []

    def __enter__(self) -> "SessionCorpus":
        self.build()
        return self

    def __exit__(self, *exc) -> bool:
        self.cleanup()
        return False

    def build(self) -> "SessionCorpus":
        if not self.paths:
            return self
        self.persist_dir = Path(tempfile.mkdtemp(prefix="syntheia-session-"))

        for path in self.paths:
            try:
                text, doc = loader.load_supplied(path)
            except Exception as e:                      # noqa: BLE001
                # A scanned PDF lands here: the loader detects zero extractable
                # text and refuses. Saying so beats indexing nothing silently
                # and leaving the user to wonder why their document did nothing.
                self.report.append((path.name, f"could not read — {e}"))
                continue

            chunks = chunker.chunk_document(text, doc)
            if not chunks:
                self.report.append((path.name, "no usable text after chunking"))
                continue
            self.chunks.extend(chunks)
            self.report.append(
                (path.name, f"{len(chunks)} passages, {len(text):,} characters"))

        if self.chunks:
            store.add_chunks(self.chunks, persist_dir=self.persist_dir)
        return self

    def cleanup(self) -> None:
        if self.persist_dir and self.persist_dir.exists():
            shutil.rmtree(self.persist_dir, ignore_errors=True)
        self.persist_dir = None

    @property
    def usable(self) -> bool:
        return bool(self.chunks and self.persist_dir)

    def summary(self) -> str:
        if not self.report:
            return "no documents supplied"
        ok = sum(1 for _, msg in self.report if "passages" in msg)
        return f"{ok} of {len(self.report)} document(s) indexed for this run"
