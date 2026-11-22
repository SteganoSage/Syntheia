"""Evidence and term matching shared by every agent that must show its source.

Two kinds of check live here, both deterministic:

1. **Is this quote really in the source?** First written for the SDLC evidence
   gate (problem log 6.5), where it stopped answers being fabricated wholesale.
   Phase B applies the same standard everywhere a model claims to be quoting:
   the conflict judge quoting two requirements, the compliance mapper quoting a
   clause, the clarification loop quoting what the user replied. One function,
   so the standard cannot drift between agents.

2. **Does this text mention one of these terms?** The rule packs
   (`data/compliance/rules.yaml`, `data/security/controls.yaml`) are written as
   term lists, and every agent that reads them must match terms the same way.

Term syntax used by the packs:

    "encrypt*"          prefix: encrypt, encrypted, encryption
    "audit trail"       phrase: whitespace and hyphens are interchangeable
    "two[- ]factor"     NOT supported -- write "two factor"; the hyphen is free
"""

from __future__ import annotations

import re
from functools import lru_cache

_WORD = re.compile(r"[a-z0-9']+")


def normalise(text: str) -> str:
    """Lowercase, straighten quotes, collapse whitespace."""
    text = text.lower().replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"').replace("—", " ")
    return re.sub(r"\s+", " ", text).strip()


def quotes_source(quote: str, source: str, min_overlap: float = 0.6) -> bool:
    """Is this quote actually traceable to the source text?

    Substring after normalisation, else a word-overlap fallback -- models
    paraphrase as often as they quote, and rejecting a faithful paraphrase would
    throw away a correct answer for a formatting reason.

    This catches invented quotes. It cannot catch a real quote attached to the
    wrong claim; that still needs a reader.
    """
    q, src = normalise(quote), normalise(source)
    if not q:
        return False
    if q in src:
        return True
    words = [w for w in _WORD.findall(q) if len(w) > 3]
    if not words:
        return False
    hits = sum(1 for w in words if w in src)
    return hits / len(words) >= min_overlap


# ── Term matching ──────────────────────────────────────────────────────

@lru_cache(maxsize=4096)
def _pattern(term: str) -> re.Pattern:
    term = term.strip().lower()
    prefix = term.endswith("*")
    if prefix:
        term = term[:-1]
    # Words separated by any run of whitespace or hyphens, so "multi factor",
    # "multi-factor" and "multi  factor" are one term.
    parts = [re.escape(p) for p in re.split(r"[\s-]+", term) if p]
    body = r"[\s-]+".join(parts)
    tail = r"\w*" if prefix else r"\b"
    return re.compile(rf"\b{body}{tail}", re.I)


def has_term(text: str, term: str) -> bool:
    return bool(text) and bool(_pattern(term).search(text))


def find_terms(text: str, terms) -> list[str]:
    """The terms (as written in the pack) that occur in the text."""
    return [t for t in terms if has_term(text, t)]


def matches_groups(text: str, groups) -> bool:
    """True when every group has at least one term in the text.

    Groups express AND-of-ORs: [["retain*", "preserv*"], ["record*"]] means
    "says retain-or-preserve AND says record". An empty group list matches
    nothing -- a rule with no way to be satisfied must not pass by default.
    """
    groups = [g for g in groups if g]
    return bool(groups) and all(find_terms(text, g) for g in groups)


_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


def sentence_with(text: str, term: str) -> str:
    """The first sentence of `text` containing `term`, for quoting as evidence."""
    for s in _SENTENCE.split(text):
        if has_term(s, term):
            return s.strip()
    return ""
