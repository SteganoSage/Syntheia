"""The SRS as a tree of sections, independent of any file format.

The builder decides WHAT the document says and in what order; renderers
(`docx_render`, `md_render`) decide only how it looks. Keeping the two apart is
what lets the same content come out as IEEE 830 or ISO/IEC/IEEE 29148, as Word
or Markdown, without four copies of the logic drifting apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Para:
    text: str
    lead: str = ""            # bold run before the text, e.g. "Note:"
    style: str = ""           # "", "note", "quote", "muted"


@dataclass
class Bullets:
    items: list[str]


@dataclass
class Table:
    headers: list[str]
    rows: list[list[str]]
    widths: list[float] | None = None     # relative column widths
    caption: str = ""


@dataclass
class Requirement:
    """One requirement, IEEE 830 §4.3 / 29148 §5.2.8 style: id, statement, attributes."""
    id: str
    statement: str
    attributes: list[tuple[str, str]]


Block = Para | Bullets | Table | Requirement


@dataclass
class Section:
    title: str
    blocks: list = field(default_factory=list)
    children: list["Section"] = field(default_factory=list)
    number: str = ""
    appendix: bool = False
    unnumbered: bool = False           # e.g. the Index: a heading, no number

    def label(self, depth: int = 1, sep: str = "   ") -> str:
        if self.unnumbered or not self.number:
            return self.title
        if self.appendix and depth == 1:
            return f"Appendix {self.number}{sep}{self.title}"
        return f"{self.number}{sep}{self.title}"

    def add(self, *blocks) -> "Section":
        self.blocks.extend(b for b in blocks if b is not None)
        return self

    def sub(self, title: str) -> "Section":
        child = Section(title)
        self.children.append(child)
        return child


@dataclass
class SRSDocument:
    title: str
    product: str
    version: str
    date: str
    status: str
    prepared_by: str
    standard: str                      # e.g. "IEEE Std 830-1998"
    layout: str                        # "ieee830" | "iso29148"
    revision_history: list[list[str]]
    approvals: list[list[str]]
    notice: str                        # what a reader must know before trusting it
    sections: list[Section] = field(default_factory=list)

    def number(self) -> "SRSDocument":
        """Assign 1, 1.1, 1.1.1 to body sections and A, B, C to appendices."""
        def walk(sec: Section, prefix: str) -> None:
            sec.number = prefix
            for i, child in enumerate(sec.children, start=1):
                walk(child, f"{prefix}.{i}")

        body = [s for s in self.sections if not s.appendix and not s.unnumbered]
        apps = [s for s in self.sections if s.appendix and not s.unnumbered]
        for i, s in enumerate(body, start=1):
            walk(s, str(i))
        for i, s in enumerate(apps):
            walk(s, chr(ord("A") + i))
        return self

    def walk(self):
        """(depth, section) for every section, in document order."""
        def go(sec: Section, depth: int):
            yield depth, sec
            for c in sec.children:
                yield from go(c, depth + 1)
        for s in self.sections:
            yield from go(s, 1)
