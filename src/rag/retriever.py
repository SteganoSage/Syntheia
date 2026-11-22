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

# Below this, the corpus has nothing on the subject and its "best" passages are
# noise. A COBOL-to-Java migration scored 0.555 and drew four loan-origination
# and digital-banking RFP passages -- the extractor then mined them and produced
# "the system shall provide a mobile-responsive online loan application portal"
# for a migration project. Scenarios the corpus does cover score 0.63-0.67.
# Ungrounded beats grounded in the wrong domain.
MIN_CONTEXT_RELEVANCE = float(os.getenv("SYNTHEIA_MIN_CONTEXT_RELEVANCE", "0.60"))
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


_REPORTED_BROKEN: set[str] = set()


def is_available(persist_dir: Path | None = None) -> bool:
    """Is there an index with anything in it? Gate RAG on this, never assume.

    Missing and broken are different answers to this question and must not look
    alike. A missing index is a fresh checkout: run ungrounded, as before. A
    store that EXISTS but errors (14 Nov: a ChromaDB compaction failure on an
    incremental build) used to return False too -- so the run printed "no
    knowledge base indexed" and went ungrounded, with the real error swallowed.
    Still False, so the run degrades rather than dies, but now it says why.
    """
    try:
        return store.stats(persist_dir)["chunks"] > 0
    except Exception as e:                       # noqa: BLE001
        target = Path(persist_dir or store.DEFAULT_DIR)
        if (target / "chroma.sqlite3").exists() and str(target) not in _REPORTED_BROKEN:
            _REPORTED_BROKEN.add(str(target))
            print(f"  !! the knowledge base at {target} exists but cannot be read "
                  f"({type(e).__name__}: {str(e)[:160]}). Running UNGROUNDED. "
                  f"Rebuild it: python -m notebooks.09_build_knowledge_base --reset")
        return False


# Words that end a regulation's name. If a requirement says "RBI's digital
# lending guidelines", the model has named a specific instrument, and that is
# checkable in a way a bare topical requirement is not.
_NAMED_REG = re.compile(
    # Parentheses are allowed in the preceding words: the Aadhaar rules are
    # cited as "Aadhaar (Authentication and Offline Verification) Regulations",
    # and without them only the bare noun matches, which carries no signal.
    r"((?:[\w'’()-]+\s+){0,5}"
    r"(?:guidelines?|regulations?|directions?|direction|act|framework|standards?|rules?|norms?)\b)",
    # The trailing \b is load-bearing. Without it "act" matched inside "account
    # activation", so a requirement to escalate high-risk customers for enhanced
    # due diligence was read as naming an instrument, found no title overlap,
    # and had a correct KYC Master Direction citation at 0.57 demoted to
    # precedent. A missing word boundary turned the safety check into the bug.
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


def _query_all(text: str, k: int, where: dict | None = None,
               persist_dir: Path | None = None,
               extra_dirs: list[Path] | None = None) -> list[dict]:
    """Search the curated corpus and any documents supplied for this run.

    One merged, re-sorted list. Supplied passages are NOT boosted -- they
    compete on the same cosine score against the same floors. Ranking them
    higher because someone chose them would be an unearned claim, and would let
    a loosely-related PDF outrank a real clause.
    """
    def one(d):
        try:
            return store.query(text, k=k, where=where, persist_dir=d)
        except Exception:
            # Metadata filtering support varies by Chroma version; an unfiltered
            # result beats no grounding at all.
            return store.query(text, k=k, persist_dir=d)

    hits = one(persist_dir)
    for d in (extra_dirs or []):
        try:
            hits += one(d)
        except Exception:
            # A broken session store must never take the curated one down with
            # it -- the run degrades to the corpus alone.
            continue
    return sorted(hits, key=lambda h: -h["score"])


def _sector_filtered(text: str, k: int, sectors: set[str] | None,
                     persist_dir, extra_dirs, where: dict | None = None) -> list[dict]:
    """`_query_all`, restricted to documents in the scenario's sector.

    Fetches deeper when filtering: the IRDAI guidelines alone are 1,262 chunks,
    and on a banking question several of the top hits can be theirs. Filtering
    the usual depth would starve the result rather than replace what was cut.
    """
    from src.rag.sectors import in_sector

    if not sectors:
        return _query_all(text, k, where, persist_dir, extra_dirs)
    return in_sector(_query_all(text, k * 3, where, persist_dir, extra_dirs), sectors)


def retrieve_context(scenario: str, k: int | None = None,
                     applicability: str | None = None,
                     persist_dir: Path | None = None,
                     extra_dirs: list[Path] | None = None,
                     sectors: set[str] | None = None) -> list[dict]:
    """Evidence for the extraction prompt.

    k defaults to 4 because of the local context budget, not because 4 is
    optimal: the extraction prompt plus scenario plus output already occupies
    roughly 1,850 of 4,096 tokens, and Day 12 measured that raising num_ctx to
    8192 on the 4 GB card spills 0.78 GB into system RAM. The cloud tier has room
    for more -- raise SYNTHEIA_CONTEXT_K there rather than re-chunking.
    """
    want = k or CONTEXT_K
    where = {"applicability": {"$contains": applicability}} if applicability else None
    hits = _sector_filtered(scenario, want * OVERFETCH, sectors, persist_dir,
                            extra_dirs, where)
    hits = _dedupe(hits, max_per_doc=MAX_PER_DOC)[:want]
    if hits and hits[0]["score"] < MIN_CONTEXT_RELEVANCE:
        return []
    return hits


def ground_requirement(statement: str, k: int | None = None,
                       min_confidence: float | None = None,
                       persist_dir: Path | None = None,
                       extra_dirs: list[Path] | None = None,
                       sectors: set[str] | None = None) -> dict:
    """Citations for one requirement. No LLM call -- embedding lookup only.

    Returns citations above the confidence floor, plus `needs_review` when
    nothing clears it: §6's "escalate unsupported or low-confidence outputs for
    human review", made concrete.
    """
    want = k or GROUND_K
    floor = MIN_CONFIDENCE if min_confidence is None else min_confidence
    # Sector comes from the SCENARIO, passed in by the pipeline: a requirement
    # statement ("the system shall assign user IDs") rarely names its sector.
    hits = _dedupe(_sector_filtered(statement, want * OVERFETCH, sectors,
                                    persist_dir, extra_dirs))

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


SUPPLIED_K = 8
MIN_SUPPLIED_RELEVANCE = 0.35


def retrieve_supplied(scenario: str, extra_dirs: list[Path],
                      k: int | None = None) -> list[dict]:
    """Passages from documents attached to THIS run, for extraction to use.

    Separate from `retrieve_context`, and held to a much lower bar, because a
    supplied document is a different kind of evidence. A corpus hit is something
    similarity search found; a supplied document is something a person chose for
    this scenario. `MIN_CONTEXT_RELEVANCE = 0.60` exists because retrieved loan
    RFPs were leaking a "mobile-responsive online loan application portal" into
    a COBOL migration -- a risk that comes from the corpus being general. It does
    not apply the same way to a file someone attached on purpose.

    So the floor here is 0.35: low enough that the relevant parts of a
    regulation reach the model, high enough to drop passages about a different
    subject entirely. More passages than the corpus gets (8 vs 4) because this
    is source material rather than phrasing reference.
    """
    if not extra_dirs:
        return []
    want = k or SUPPLIED_K
    hits: list[dict] = []
    for d in extra_dirs:
        try:
            hits += store.query(scenario, k=want * 2, persist_dir=d)
        except Exception:
            continue
    hits = sorted(hits, key=lambda h: -h["score"])
    return [h for h in hits if h["score"] >= MIN_SUPPLIED_RELEVANCE][:want]


def format_supplied(hits: list[dict]) -> str:
    """Render attached documents as SOURCE MATERIAL, not as reference.

    The opposite instruction to `format_context`, on purpose. Corpus passages
    are reference only -- the model must not take requirements from them,
    because nobody asserted they apply. These were attached for this project by
    the person running it, which is the assertion, and deriving requirements
    from them is the supervisor's part 1 in its most literal form: "feed a
    lightweight LLM existing requirement documents... then prompt it to generate
    the specific requirements".

    The guard that remains is relevance to the SYSTEM the scenario describes. A
    regulation covers a whole industry; only the parts that bear on this system
    are requirements of this project.
    """
    if not hits:
        return ""
    blocks = []
    for i, h in enumerate(hits, 1):
        m = h["metadata"]
        label = m.get("citation") or m.get("title", "unknown source")
        blocks.append(f"[D{i}] {label}\n{h['text'].strip()}")
    return (
        "### Documents provided for this project\n\n"
        "The passages below come from documents supplied ALONGSIDE the scenario "
        "for this specific project. Treat them as source material, not as "
        "background.\n\n"
        "**Derive requirements from them.** Where a passage states an "
        "obligation, a limit, a retention period or a control that applies to "
        "the system the scenario describes, that is a requirement of this "
        "project even though the scenario does not spell it out. Cite the "
        "passage in `applicable_regulations` when you do.\n\n"
        "Two limits:\n"
        "- The scenario still defines the project. A passage about a capability "
        "this system does not have is not a requirement of it -- a rule about "
        "motor claims does not belong in a health-claims portal.\n"
        "- Prefer the document's own numbers. If it says ten years, or five "
        "steps, or 99.5 per cent, use that figure rather than inventing one.\n\n"
        + "\n\n".join(blocks)
    )


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
        "### Reference material\n\n"
        "The passages below are extracts from finance regulations and from other "
        "institutions' requirement documents. They are REFERENCE ONLY.\n\n"
        "Every requirement you produce must come from the problem scenario. "
        "Do NOT take requirements from these passages: if a passage describes a "
        "feature the scenario does not ask for, it is not a requirement of this "
        "project. Use them only to phrase requirements the way this domain "
        "phrases them, and to identify which regulations apply.\n\n"
        + "\n\n".join(blocks)
    )
