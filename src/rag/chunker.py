"""Split documents into retrievable units along their own structure.

Fixed-size chunking (512 tokens, 50 overlap) is the default everywhere and is the
wrong default here. Regulations and requirement specifications are *numbered*:
the unit a compliance officer checks is "RBI KYC Master Direction §16(b)", and
the unit a requirements engineer checks is "FR-042". A fixed window cuts across
those boundaries, so the best citation the pipeline can offer becomes "somewhere
in this PDF" -- which is not a citation.

So: split on headings and clause numbers, and fall back to size-based splitting
only inside a section too large to embed usefully. A chunk then carries the
clause id it came from, and `Chunk.citation` resolves to something checkable.

The heading patterns below are ordered most- to least- specific. They cover what
the corpus actually contains:
    9.  / 9.1 / 9.1.3          numbered sections (regulations, SRS)
    (a) (b) (iv)               sub-clauses (RBI master directions use these)
    Section 5 / Clause 12      spelled-out references
    Chapter VI / Part III      roman-numbered divisions (RBI, Basel)
    REQ-001 / FR-12 / NFR-3    requirement ids (SRS, tender documents)
"""

from __future__ import annotations

import re

from src.rag.schema import Chunk, SourceDoc

# Target chunk size in characters. ~1200 chars is roughly 300 tokens, which keeps
# 3-5 chunks inside the 4096-token local window alongside prompt and output. The
# cloud tier has room for more; the chunk size stays the same so retrieval is
# identical across tiers and only the number injected changes.
TARGET_CHARS = 1200
MAX_CHARS = 2000
MIN_CHARS = 120          # below this a "chunk" is a heading or page furniture

_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("req",     re.compile(r"^\s*((?:REQ|FR|NFR|UC|BR)[-–\s]?\d+(?:\.\d+)*)\b[.:)\s]", re.I)),
    ("chapter", re.compile(r"^\s*(?:CHAPTER|PART|ANNEX(?:URE)?|SCHEDULE)\s+([IVXLC]+|\d+)\b", re.I)),
    ("named",   re.compile(r"^\s*(?:SECTION|CLAUSE|ARTICLE|RULE|REGULATION)\s+(\d+(?:\.\d+)*)\b", re.I)),
    ("numbered", re.compile(r"^\s*(\d+(?:\.\d+){0,3})\.?\s+(?=[A-Z(\"'])")),
    ("alpha",   re.compile(r"^\s*\(([a-z]{1,2}|[ivxl]{1,5})\)\s+")),
]


def detect_heading(line: str) -> tuple[str, str] | None:
    """(kind, label) when a line opens a new structural unit, else None."""
    stripped = line.strip()
    if not stripped or len(stripped) > 200:
        # A very long line is body text that happens to start with a number,
        # not a heading. This one rule removes most false splits.
        return None
    for kind, pattern in _PATTERNS:
        m = pattern.match(stripped)
        if m:
            return kind, m.group(1)
    return None


def _split_oversized(text: str) -> list[str]:
    """Break a too-large section on paragraph, then sentence boundaries."""
    if len(text) <= MAX_CHARS:
        return [text]
    out, buf = [], ""
    for para in re.split(r"\n\s*\n", text):
        if len(buf) + len(para) + 2 <= TARGET_CHARS or not buf:
            buf = f"{buf}\n\n{para}".strip() if buf else para
        else:
            out.append(buf)
            buf = para
        while len(buf) > MAX_CHARS:                   # single enormous paragraph
            cut = buf.rfind(". ", 0, MAX_CHARS)
            cut = cut + 1 if cut > MIN_CHARS else MAX_CHARS
            out.append(buf[:cut].strip())
            buf = buf[cut:].strip()
    if buf:
        out.append(buf)
    return [c for c in out if c.strip()]


def _level(kind: str, label: str, numeric_level: int) -> int:
    """Nesting level, 0-based, so heading_path truncates correctly.

    "9" is level 1, "9.2" level 2, "9.2.1" level 3; a chapter sits above all of
    them at 0; a lettered sub-clause sits one below whatever numbered section is
    currently open. Getting this wrong makes the breadcrumb accumulate siblings
    instead of ancestors -- "9.1 > a > b > c" rather than "9 > 9.2 > (b)".
    """
    if kind == "chapter":
        return 0
    if kind == "alpha":
        return max(numeric_level + 1, 2)
    if kind == "req":
        return 1
    return 1 + label.count(".")


def _clause_id(stack: list[tuple[int, str, str]]) -> str:
    """Compose a citable id from the open heading stack.

    A bare "(a)" cites nothing -- there is an (a) in every section. The id has to
    carry its numbered parent, so a compliance officer can find it: 9.2(a).
    """
    numeric = ""
    letters: list[str] = []
    for _, kind, label in stack:
        if kind in {"numbered", "named", "req"}:
            numeric, letters = label, []
        elif kind == "alpha":
            letters.append(label)
    return numeric + "".join(f"({l})" for l in letters)


def chunk_document(text: str, doc: SourceDoc) -> list[Chunk]:
    """Structural chunks for one document, each citable by clause."""
    lines = text.splitlines(keepends=True)

    starts: list[tuple[int, str, str]] = []
    for i, line in enumerate(lines):
        hit = detect_heading(line)
        if hit:
            starts.append((i, hit[0], hit[1]))

    if not starts:
        # Unstructured document, or a PDF whose numbering did not survive
        # extraction. Size-based splitting is the honest fallback, and clause_id
        # stays empty so the citation claims no precision it does not have.
        return _renumber(_as_chunks(_split_oversized(text), doc, "", [], text), doc)

    stack: list[tuple[int, str, str]] = []     # (level, kind, label)
    numeric_level = 1
    pending = ""                               # short lead-in carried forward
    sections: list[tuple[str, list[str], str]] = []

    for n, (line_idx, kind, label) in enumerate(starts):
        end_idx = starts[n + 1][0] if n + 1 < len(starts) else len(lines)
        body = "".join(lines[line_idx:end_idx]).strip()

        level = _level(kind, label, numeric_level)
        if kind in {"numbered", "named", "req"}:
            numeric_level = level
        stack = [e for e in stack if e[0] < level] + [(level, kind, label)]

        if len(body) < MIN_CHARS:
            # A heading or a one-line lead-in. Do NOT drop it: the stack must
            # still advance (or the breadcrumb loses an ancestor) and the text
            # belongs with what follows -- "shall obtain the following:" is
            # meaningless separated from the list it introduces.
            pending = f"{pending}\n{body}".strip() if pending else body
            continue

        if pending:
            body = f"{pending}\n{body}"
            pending = ""
        sections.append((_clause_id(stack), [e[2] for e in stack], body))

    if pending and sections:
        sections[-1] = (sections[-1][0], sections[-1][1],
                        f"{sections[-1][2]}\n{pending}")

    chunks: list[Chunk] = []
    for clause_id, heading_path, body in sections:
        for piece in _split_oversized(body):
            chunks.extend(_as_chunks([piece], doc, clause_id, heading_path, text))
    return _renumber(chunks, doc)


def _as_chunks(pieces, doc, clause_id, heading_path, full_text) -> list[Chunk]:
    out = []
    for piece in pieces:
        if len(piece.strip()) < MIN_CHARS:
            continue
        start = full_text.find(piece[:80])
        out.append(Chunk(
            chunk_id="", text=piece.strip(), doc=doc, clause_id=clause_id,
            heading_path=heading_path,
            char_start=max(start, 0), char_end=max(start, 0) + len(piece),
        ))
    return out


def _renumber(chunks: list[Chunk], doc: SourceDoc) -> list[Chunk]:
    stem = doc.path.rsplit("/", 1)[-1].rsplit(".", 1)[0][:48]
    for i, c in enumerate(chunks):
        c.chunk_id = f"{stem}::{i:04d}"
    return chunks
