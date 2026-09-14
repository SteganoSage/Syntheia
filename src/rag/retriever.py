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
import re
from pathlib import Path

from src.rag import store

# Raised from 0.45 to 0.55 after the first grounded run on the real corpus: at
# 0.45 a CIBIL credit-score requirement drew citations scoring 0.48, which were
# noise. Still not properly calibrated -- and worth being clear that a threshold
# cannot fix the harder failure. On that same run, "comply with RBI's digital
# lending guidelines" cited the RBI IT Governance direction at 0.67, because the
# Digital Lending Guidelines are not in the corpus and that was the nearest
# RBI-shaped document. No confidence floor distinguishes "right document" from
# "plausible wrong document"; only corpus coverage does.
MIN_CONFIDENCE = float(os.getenv("SYNTHEIA_MIN_CONFIDENCE", "0.55"))

CONTEXT_K = int(os.getenv("SYNTHEIA_CONTEXT_K", "4"))
GROUND_K = int(os.getenv("SYNTHEIA_GROUND_K", "3"))

# Retrieval over-fetches by this factor, then deduplicates and caps per
# document. Without it the first grounded run returned four passages for a loan
# scenario that were all from one bank's digital-banking RFP -- with clause §181
# appearing twice, wasting a quarter of a four-slot context budget.
OVERFETCH = 4
MAX_PER_DOC = int(os.getenv("SYNTHEIA_MAX_PER_DOC", "2"))

# Only these document types may populate `applicable_regulations`. A tender is
# not a regulation: citing another bank's procurement RFP in a field named
# "applicable regulations" is a category error, however good the similarity
# score. Requirement documents earn their place in the corpus by teaching the
# extractor how requirements in this domain are written -- that happens through
# retrieve_context, not by being mislabelled as law.
REGULATORY_TYPES = {"regulation", "standard"}


def is_available(persist_dir: Path | None = None) -> bool:
    """Is there an index with anything in it? Gate RAG on this, never assume."""
    try:
        return store.stats(persist_dir)["chunks"] > 0
    except Exception:                            # noqa: BLE001
        return False


# Words that end a regulation's name. If a requirement says "RBI's digital
# lending guidelines", the model has named a specific instrument, and that is
# checkable in a way a bare topical requirement is not.
_NAMED_REG = re.compile(
    # Parentheses are allowed in the preceding words: the Aadhaar rules are
    # cited as "Aadhaar (Authentication and Offline Verification) Regulations",
    # and without them only the bare noun matches, which carries no signal.
    r"((?:[\w'’()-]+\s+){0,5}"
    r"(?:guidelines?|regulations?|directions?|direction|act|framework|standards?|rules?|norms?))",
    re.I,
)

# Dropped when comparing a named instrument against a citation: they carry no
# distinguishing signal. Issuer acronyms are excluded deliberately -- every RBI
# document matches "RBI", so leaving them in would let any RBI direction satisfy
# any named RBI rule, which is the exact failure this check exists to catch.
_STOP = {
    "the", "of", "and", "for", "on", "in", "with", "a", "an", "to", "s",
    "guidelines", "guideline", "regulations", "regulation", "directions",
    "direction", "act", "framework", "standards", "standard", "rules", "rule",
    "norms", "norm", "master", "shall", "system", "comply", "compliance",
    "rbi", "sebi", "irdai", "uidai", "npci", "reserve", "bank", "india",
}


def _named_instruments(statement: str) -> list[set[str]]:
    """Distinctive word-sets for each regulation the statement names by name."""
    out = []
    for phrase in _NAMED_REG.findall(statement):
        words = re.split(r"[\s/,()]+", phrase)
        # Strip the possessive before stopword removal, or "RBI's" survives a
        # list that contains "rbi" and becomes a distinguishing token it is not.
        tokens = {re.sub(r"['’]s$", "", w.lower()).strip("'’-") for w in words}
        tokens = {w for w in tokens - _STOP if len(w) > 2}
        if tokens:
            out.append(tokens)
    return out


def _supports_named(citations: list[dict], statement: str) -> bool:
    """Does any citation plausibly *be* the instrument the statement names?

    Similarity retrieval answers "what is about this subject", not "is this the
    document you asked for". On the first grounded run a requirement to comply
    with RBI's digital lending guidelines -- absent from the corpus -- drew the
    RBI IT Governance direction at 0.67 and two Prepaid Payment Instrument
    clauses. All three are real RBI regulations about nothing of the sort.

    So when a statement names an instrument, at least one distinctive word from
    that name must appear in a cited document's title. No overlap means the
    corpus does not contain what was asked for, whatever the cosine score says.
    """
    named = _named_instruments(statement)
    if not named:
        return True                       # nothing named; nothing to check
    haystack = " ".join(c.get("title", "") for c in citations).lower()
    return any(any(tok in haystack for tok in tokens) for tokens in named)


def _dedupe(hits: list[dict], max_per_doc: int | None = None) -> list[dict]:
    """Drop repeats of the same clause, and optionally cap per document.

    Chunks from one clause can be near-identical after oversized-section
    splitting, so the same §44 comes back twice with the same score. Each slot
    spent on a duplicate is a slot not spent on a different source.
    """
    seen: set[tuple[str, str]] = set()
    per_doc: dict[str, int] = {}
    out = []
    for h in hits:
        m = h["metadata"]
        key = (m.get("doc_path", ""), m.get("clause_id", ""))
        if key in seen:
            continue
        doc = m.get("doc_path", "")
        if max_per_doc is not None and per_doc.get(doc, 0) >= max_per_doc:
            continue
        seen.add(key)
        per_doc[doc] = per_doc.get(doc, 0) + 1
        out.append(h)
    return out


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
    want = k or CONTEXT_K
    where = {"applicability": {"$contains": applicability}} if applicability else None
    try:
        hits = store.query(scenario, k=want * OVERFETCH, where=where, persist_dir=persist_dir)
    except Exception:
        # Metadata filtering support varies by Chroma version; an unfiltered
        # result is far better than no grounding at all.
        hits = store.query(scenario, k=want * OVERFETCH, persist_dir=persist_dir)
    return _dedupe(hits, max_per_doc=MAX_PER_DOC)[:want]


def ground_requirement(statement: str, k: int | None = None,
                       min_confidence: float | None = None,
                       persist_dir: Path | None = None) -> dict:
    """Citations for one requirement. No LLM call -- embedding lookup only.

    Returns citations above the confidence floor, plus `needs_review` when
    nothing clears it: §6's "escalate unsupported or low-confidence outputs for
    human review", made concrete.
    """
    want = k or GROUND_K
    floor = MIN_CONFIDENCE if min_confidence is None else min_confidence
    hits = _dedupe(store.query(statement, k=want * OVERFETCH, persist_dir=persist_dir))

    def as_citation(h: dict) -> dict:
        m = h["metadata"]
        return {
            "citation": m.get("citation", ""),
            "clause_id": m.get("clause_id", ""),
            "title": m.get("title", ""),
            "issuer": m.get("issuer", ""),
            "version": m.get("version", ""),
            "doc_type": m.get("doc_type", ""),
            "url": m.get("url", ""),
            "confidence": h["score"],
            "excerpt": h["text"][:300],
        }

    above = [h for h in hits if h["score"] >= floor]
    # Split by what the source actually is. Regulations bind; another bank's
    # tender only shows how someone else phrased a similar requirement. Both are
    # useful, and conflating them is what put procurement RFPs into a field
    # named "applicable regulations" on the first grounded run.
    citations = [as_citation(h) for h in above
                 if h["metadata"].get("doc_type") in REGULATORY_TYPES][:want]
    precedents = [as_citation(h) for h in above
                  if h["metadata"].get("doc_type") not in REGULATORY_TYPES][:want]

    # A named instrument that nothing in the corpus matches is unsupported, no
    # matter how confidently something similar scored. Keep the near-misses as
    # precedent rather than discarding them -- they are still the closest thing
    # the corpus holds, and a reviewer will want to see what was considered.
    unnamed_mismatch = bool(citations) and not _supports_named(citations, statement)
    if unnamed_mismatch:
        precedents = citations + precedents
        citations = []

    best = max((h["score"] for h in hits), default=0.0)
    return {
        "citations": citations,
        "named_instrument_unmatched": unnamed_mismatch,
        "applicable_regulations": [c["citation"] for c in citations],
        # How comparable systems specified this. Evidence for the reasoning the
        # supervisor's part 1 asks for, not a compliance claim.
        "precedents": precedents,
        "confidence": round(best, 4),
        # No *regulation* supports this requirement. The model may still have
        # named one; it is simply unverified against the corpus, which is the
        # case §6 says to escalate.
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
