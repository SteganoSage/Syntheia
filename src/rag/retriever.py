"""The two retrieval calls the pipeline makes.

Problem statement §6 gives the workflow: retrieve relevant policies and domain
documents, provide the evidence to the LLM, generate, attach citations and
confidence, escalate what is unsupported. That maps onto two distinct calls with
very different costs:

    retrieve_context(scenario)     once per run, feeds the extraction prompt
    ground_requirement(statement)  once per requirement, attaches citations

The second deliberately makes **no LLM call**. A second generation pass per
requirement would cost seconds each on a 3B model; an embedding lookup costs
about 40 ms, and produces a citation that points at a clause. The trade is that
similarity is not the same as applicability -- a high cosine score means the text
is *about* the same subject, not that the rule *applies*. That is exactly why the
score is surfaced as a confidence and why low-confidence citations are flagged
for human review rather than asserted.

MIN_CONFIDENCE below is a placeholder, not a calibrated value. Calibrate it on
Day 14 against real documents the way the classifier threshold was calibrated:
sweep it, look at what it admits and rejects, and record the number.
"""

from __future__ import annotations

import os
from pathlib import Path

from src.rag import store

# Uncalibrated. On the synthetic fixture, correct hits scored 0.50-0.70 and the
# first wrong one 0.37 -- so 0.45 separates them there. Real regulatory prose
# will differ; treat this as a starting point to measure, never as tuned.
MIN_CONFIDENCE = float(os.getenv("SYNTHEIA_MIN_CONFIDENCE", "0.45"))

CONTEXT_K = int(os.getenv("SYNTHEIA_CONTEXT_K", "4"))
GROUND_K = int(os.getenv("SYNTHEIA_GROUND_K", "3"))


def is_available(persist_dir: Path | None = None) -> bool:
    """Is there an index with anything in it? Gate RAG on this, never assume."""
    try:
        return store.stats(persist_dir)["chunks"] > 0
    except Exception:                            # noqa: BLE001
        return False


def retrieve_context(scenario: str, k: int | None = None,
                     applicability: str | None = None,
                     persist_dir: Path | None = None) -> list[dict]:
    """Evidence for the extraction prompt.

    k defaults to 4 because of the local context budget, not because 4 is
    optimal: the extraction prompt plus scenario plus output already occupies
    roughly 1,850 of 4,096 tokens, and Day 12 measured that raising num_ctx to
    8192 on the 4 GB card spills 0.78 GB into system RAM. The cloud tier has room
    for more -- raise SYNTHEIA_CONTEXT_K there rather than re-chunking.
    """
    where = {"applicability": {"$contains": applicability}} if applicability else None
    try:
        return store.query(scenario, k=k or CONTEXT_K, where=where, persist_dir=persist_dir)
    except Exception:
        # Metadata filtering support varies by Chroma version; an unfiltered
        # result is far better than no grounding at all.
        return store.query(scenario, k=k or CONTEXT_K, persist_dir=persist_dir)


def ground_requirement(statement: str, k: int | None = None,
                       min_confidence: float | None = None,
                       persist_dir: Path | None = None) -> dict:
    """Citations for one requirement. No LLM call -- embedding lookup only.

    Returns citations above the confidence floor, plus `needs_review` when
    nothing clears it: §6's "escalate unsupported or low-confidence outputs for
    human review", made concrete.
    """
    floor = MIN_CONFIDENCE if min_confidence is None else min_confidence
    hits = store.query(statement, k=k or GROUND_K, persist_dir=persist_dir)

    citations = [
        {
            "citation": h["metadata"].get("citation", ""),
            "clause_id": h["metadata"].get("clause_id", ""),
            "title": h["metadata"].get("title", ""),
            "issuer": h["metadata"].get("issuer", ""),
            "version": h["metadata"].get("version", ""),
            "url": h["metadata"].get("url", ""),
            "confidence": h["score"],
            "excerpt": h["text"][:300],
        }
        for h in hits if h["score"] >= floor
    ]
    best = max((h["score"] for h in hits), default=0.0)
    return {
        "citations": citations,
        "applicable_regulations": [c["citation"] for c in citations],
        "confidence": round(best, 4),
        # Nothing cleared the floor: the model may still name a regulation, but
        # no source in the knowledge base supports it. That is the case a human
        # must look at.
        "needs_review": not citations,
    }


def format_context(hits: list[dict]) -> str:
    """Render retrieved evidence for a prompt.

    Each passage is labelled with its citation so the model can attribute rather
    than paraphrase anonymously, and the instruction is explicit that inventing
    a source is worse than admitting none -- the behaviour §6 exists to force.
    """
    if not hits:
        return ""
    blocks = []
    for i, h in enumerate(hits, 1):
        m = h["metadata"]
        label = m.get("citation") or m.get("title", "unknown source")
        blocks.append(f"[E{i}] {label}\n{h['text'].strip()}")
    return (
        "### Retrieved source material\n\n"
        "The passages below are from approved finance documents. Ground the "
        "requirements you produce in them, and cite the [E#] label of any "
        "passage you rely on. If none of them supports a point, say so rather "
        "than naming a regulation they do not contain.\n\n"
        + "\n\n".join(blocks)
    )
