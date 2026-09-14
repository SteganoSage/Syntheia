"""Read knowledge-base documents into text plus provenance.

Provenance cannot be inferred from a PDF. Publishers' embedded metadata is
usually blank, wrong, or says "Microsoft Word - final_v3.docx". So each document
carries a hand-written sidecar manifest, and loading **fails loudly** when one is
missing rather than ingesting an unciteable document:

    rbi_kyc_master_direction.pdf
    rbi_kyc_master_direction.manifest.yaml

Two minutes of typing per document buys every chunk from it a citation that can
be checked. A document without provenance is worse than no document -- it lets
the pipeline produce confident references nobody can verify, which is the
failure §6 exists to prevent.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from src.rag.schema import SourceDoc

SUPPORTED = {".pdf", ".docx", ".md", ".txt"}


class MissingManifest(FileNotFoundError):
    """Raised when a document has no sidecar. Deliberately fatal."""


def manifest_path_for(doc_path: Path) -> Path:
    return doc_path.with_suffix(doc_path.suffix + ".manifest.yaml")


def load_manifest(doc_path: Path) -> SourceDoc:
    mpath = manifest_path_for(doc_path)
    if not mpath.exists():
        raise MissingManifest(
            f"No manifest for {doc_path.name}.\n"
            f"Create {mpath.name} with at least: title, issuer, jurisdiction.\n"
            f"Problem statement §5 requires source, jurisdiction, effective date, "
            f"version and applicability on every knowledge item -- without them a "
            f"retrieved passage cannot be cited."
        )
    data = yaml.safe_load(mpath.read_text(encoding="utf-8")) or {}
    data["path"] = str(doc_path)
    return SourceDoc(**data)


# ── Text extraction ────────────────────────────────────────────────────

def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [(p.extract_text() or "") for p in reader.pages]
    text = "\n\n".join(pages)
    if len(text.strip()) < 200 * len(pages) ** 0.5:
        # A scanned PDF yields almost nothing per page. Say so plainly rather
        # than silently indexing empty chunks -- OCR is a decision, not a default.
        raise ValueError(
            f"{path.name}: extracted only {len(text.strip())} chars from "
            f"{len(pages)} pages. This is probably a scanned image PDF and needs "
            f"OCR before it can be used. Skipping is usually the right call."
        )
    return text


def _read_docx(path: Path) -> str:
    import docx

    d = docx.Document(str(path))
    parts = [p.text for p in d.paragraphs]
    for table in d.tables:                       # requirement tables in tenders
        for row in table.rows:
            parts.append(" | ".join(c.text.strip() for c in row.cells))
    return "\n".join(parts)


def read_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(path)
    if suffix == ".docx":
        return _read_docx(path)
    if suffix in {".md", ".txt"}:
        return path.read_text(encoding="utf-8", errors="replace")
    raise ValueError(f"Unsupported file type {suffix} for {path.name}")


def load_document(path: Path) -> tuple[str, SourceDoc]:
    """Text and provenance for one document. Raises if either is unusable."""
    doc = load_manifest(path)
    return read_text(path), doc


def discover(root: Path) -> list[Path]:
    """Every ingestable document under `root`, manifests themselves excluded."""
    return sorted(
        p for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in SUPPORTED
        and not p.name.endswith(".manifest.yaml")
    )
