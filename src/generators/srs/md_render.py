"""Render an SRSDocument to Markdown -- for diffs, review in git, and quick reading."""

from __future__ import annotations

from src.generators.srs.model import Bullets, Para, Requirement, SRSDocument, Table


def _cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", "<br>")


def _table(t: Table) -> list[str]:
    out = []
    if t.caption:
        out.append(f"*{t.caption}*\n")
    out.append("| " + " | ".join(_cell(h) for h in t.headers) + " |")
    out.append("|" + "|".join("---" for _ in t.headers) + "|")
    out += ["| " + " | ".join(_cell(v) for v in row) + " |" for row in t.rows]
    return out + [""]


def _requirement(r: Requirement) -> list[str]:
    out = [f"**{r.id}** — {r.statement}", "", "| Attribute | Value |", "|---|---|"]
    out += [f"| {_cell(k)} | {_cell(v)} |" for k, v in r.attributes]
    return out + [""]


def render_markdown(srs: SRSDocument) -> str:
    out = [f"# {srs.title}", f"## for {srs.product}", "",
           f"**Version:** {srs.version}  ", f"**Date:** {srs.date}  ",
           f"**Status:** {srs.status}  ", f"**Prepared by:** {srs.prepared_by}  ",
           f"**Conforms to:** {srs.standard}", "", f"> {srs.notice}", "",
           "### Revision history", ""]
    out += _table(Table(["Version", "Date", "Description", "Author"], srs.revision_history))
    out += ["### Approvals", ""]
    out += _table(Table(["Role", "Name", "Signature", "Date"], srs.approvals))
    out += ["### Table of contents", ""]
    for depth, sec in srs.walk():
        if depth <= 3:
            out.append(f"{'  ' * (depth - 1)}- {sec.label(depth, ' ')}")
    out.append("")
    for depth, sec in srs.walk():
        out += [f"{'#' * min(depth + 1, 6)} {sec.label(depth, ' — ' if sec.appendix and depth == 1 else ' ')}", ""]
        for b in sec.blocks:
            if isinstance(b, Requirement):
                out += _requirement(b)
            elif isinstance(b, Table):
                out += _table(b)
            elif isinstance(b, Bullets):
                out += [f"- {i}" for i in b.items] + [""]
            elif isinstance(b, Para):
                text = f"**{b.lead}** {b.text}" if b.lead else b.text
                out += [f"*{text}*" if b.style == "muted" else
                        f"> {text}" if b.style == "quote" else text, ""]
    return "\n".join(out)
