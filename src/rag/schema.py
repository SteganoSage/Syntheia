"""Types for the financial knowledge base.

Problem statement §5 requires that "every knowledge item should include its
source, jurisdiction, effective date, version, and applicability." That is not
bureaucracy: in a regulated domain a citation is only worth something if you can
say *which version of which rule* a requirement was checked against. A retrieved
paragraph with no provenance is an unverifiable claim, which is exactly what §6
exists to prevent.

So provenance is required at load time and travels with every chunk, rather than
being attached later when the information has been lost.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

DocType = Literal["regulation", "srs", "api_spec", "tender", "standard", "glossary", "template"]

# Which parts of the business a document speaks to. Used to filter retrieval --
# a lending scenario should not be grounded in insurance solvency rules.
APPLICABILITY = [
    "kyc_onboarding",
    "lending",
    "payments",
    "fraud_detection",
    "insurance",
    "regulatory_reporting",
    "digital_banking",
    "general",
]


class SourceDoc(BaseModel):
    """One document in the knowledge base, with the provenance §5 demands."""

    path: str
    title: str
    issuer: str = Field(description="Organisation that published it, e.g. 'Reserve Bank of India'")
    doc_id: str = Field(default="none", description="Official reference, e.g. RBI/DBR/2015-16/18")
    jurisdiction: str = Field(description="IN, UK, EU, US, GLOBAL")
    effective_date: date | None = None
    version: str = Field(default="", description="Version or amendment label")
    applicability: list[str] = Field(default_factory=lambda: ["general"])
    doc_type: DocType = "regulation"
    url: str = ""
    # "curated"  -- in data/knowledge_base, with a hand-written manifest giving
    #               the source, jurisdiction and effective date §5 requires
    # "supplied" -- handed to a single run by whoever was using it. No manifest
    #               exists, so none of that provenance can be claimed, and a
    #               citation drawn from it must say so rather than look like a
    #               curated one.
    provenance: str = "curated"

    def citation(self, clause_id: str = "") -> str:
        """Human-readable citation, the string that ends up in a requirement.

        A supplied document is marked. It has no manifest, so its issuer,
        jurisdiction and effective date are unknown -- and a citation that looks
        identical to one from the curated corpus would quietly claim provenance
        the system cannot vouch for. That is the specific thing §5 exists to
        prevent, so the difference is visible in the citation string itself
        rather than in a field someone has to go and check.
        """
        bits = ["[supplied]"] if self.provenance != "curated" else []
        bits.append(self.title)
        if clause_id:
            bits.append(f"§{clause_id}")
        if self.doc_id and self.doc_id != "none":
            bits.append(f"({self.doc_id})")
        if self.version:
            bits.append(f"[{self.version}]")
        return " ".join(bits)


class Chunk(BaseModel):
    """A retrievable unit of text plus everything needed to cite it.

    `clause_id` is the reason chunking is structural rather than fixed-size: a
    citation that reads "§16(b)" can be checked by a compliance officer, one that
    reads "characters 2048-2560" cannot.
    """

    chunk_id: str
    text: str
    doc: SourceDoc
    clause_id: str = ""
    heading_path: list[str] = Field(default_factory=list)
    char_start: int = 0
    char_end: int = 0

    @property
    def citation(self) -> str:
        return self.doc.citation(self.clause_id)

    def metadata(self) -> dict:
        """Flat metadata for ChromaDB -- it accepts only scalars, so lists are joined."""
        return {
            "doc_path": self.doc.path,
            "title": self.doc.title,
            "issuer": self.doc.issuer,
            "doc_id": self.doc.doc_id,
            "jurisdiction": self.doc.jurisdiction,
            "effective_date": self.doc.effective_date.isoformat() if self.doc.effective_date else "",
            "version": self.doc.version,
            "doc_type": self.doc.doc_type,
            "applicability": ",".join(self.doc.applicability),
            "url": self.doc.url,
            "clause_id": self.clause_id,
            "heading_path": " > ".join(self.heading_path),
            "citation": self.citation,
        }
