"""Render an SRSDocument to Word (.docx) with python-docx.

Layout choices a reviewer expects of a formal SRS:
  * title page; document-control page (revision history, sign-off table for
    the approvers PS §16 names); table of contents; then the numbered body
  * headings use Word's built-in Heading 1-4 styles -- the table-of-contents
    field only collects those -- and each top-level section starts a new page
  * every requirement is a shaded attribute table that will not split across
    a page, with its id and "shall" statement as the header row
  * running header with the document title; footer "Page X of Y"

The table of contents and page numbers are Word fields. They are marked to
update when the file is opened, so Word fills them in on first open.
"""

from __future__ import annotations

import re

from docx import Document
from docx.enum.section import WD_ORIENT  # noqa: F401  (kept for future landscape annexes)
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

from src.generators.srs.model import Bullets, Para, Requirement, SRSDocument, Table

FONT = "Calibri"
NAVY = RGBColor(0x1F, 0x38, 0x64)
GREY = RGBColor(0x59, 0x59, 0x59)
HEADER_FILL = "D9E2F3"
LABEL_FILL = "F2F2F2"
REQ_FILL = "DEEAF6"
PAGE_W, MARGIN = 21.0, 2.2                        # A4, cm
TEXT_W = PAGE_W - 2 * MARGIN                      # 16.6 cm usable


# ── Low-level helpers ──────────────────────────────────────────────────

def _shade(cell, fill: str) -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")                 # never "solid": renders black
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tcPr.append(shd)


def _field(paragraph, instr: str, placeholder: str = "") -> None:
    def fld(kind):
        r = paragraph.add_run()
        el = OxmlElement("w:fldChar")
        el.set(qn("w:fldCharType"), kind)
        if kind == "begin":
            el.set(qn("w:dirty"), "true")
        r._r.append(el)

    fld("begin")
    r = paragraph.add_run()
    it = OxmlElement("w:instrText")
    it.set(qn("xml:space"), "preserve")
    it.text = f" {instr} "
    r._r.append(it)
    fld("separate")
    paragraph.add_run(placeholder)
    fld("end")


def _repeat_header(row) -> None:
    trPr = row._tr.get_or_add_trPr()
    el = OxmlElement("w:tblHeader")
    el.set(qn("w:val"), "true")
    trPr.append(el)


def _cant_split(row) -> None:
    trPr = row._tr.get_or_add_trPr()
    trPr.append(OxmlElement("w:cantSplit"))


_ID = re.compile(r"\b(?:REQ|RISK|CMP|SEC|PRIV|CQ|TBD|UC)(?:-[A-Z0-9]+)+\b")


def _nobreak_ids(text: str) -> str:
    """REQ-001 or CMP-AADHAAR-01 must not wrap at a hyphen: use U+2011 throughout."""
    return _ID.sub(lambda m: m.group(0).replace("-", "\u2011"), text)


def _cell_text(cell, text: str, bold: bool = False, size: float = 9.5,
               color: RGBColor | None = None) -> None:
    text = _nobreak_ids(text)
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(1)
    p.paragraph_format.space_before = Pt(1)
    run = p.add_run(text)
    run.bold = bold
    run.font.size = Pt(size)
    if color:
        run.font.color.rgb = color


def _widths(table, rel: list[float] | None, total: float = TEXT_W) -> None:
    n = len(table.columns)
    rel = rel or [1.0] * n
    s = sum(rel)
    cms = [total * w / s for w in rel]
    table.autofit = False
    for row in table.rows:
        for cell, w in zip(row.cells, cms):
            cell.width = Cm(w)


# ── Styles and page ────────────────────────────────────────────────────

def _setup(doc) -> None:
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    for side in ("left_margin", "right_margin"):
        setattr(sec, side, Cm(MARGIN))
    sec.top_margin, sec.bottom_margin = Cm(2.3), Cm(2.0)
    sec.different_first_page_header_footer = True

    normal = doc.styles["Normal"]
    normal.font.name = FONT
    normal.font.size = Pt(11)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.12

    sizes = {1: 16, 2: 13, 3: 11.5, 4: 11}
    before = {1: 0, 2: 14, 3: 10, 4: 8}
    for lvl, size in sizes.items():
        st = doc.styles[f"Heading {lvl}"]
        st.font.name = FONT
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.italic = False
        st.font.color.rgb = NAVY
        st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
        pf = st.paragraph_format
        pf.space_before = Pt(before[lvl])
        pf.space_after = Pt(6 if lvl == 1 else 4)
        pf.keep_with_next = True
        pf.page_break_before = (lvl == 1)

    for name in ("List Bullet",):
        doc.styles[name].font.name = FONT
        doc.styles[name].font.size = Pt(10.5)


def _header_footer(doc, srs: SRSDocument) -> None:
    sec = doc.sections[0]
    hp = sec.header.paragraphs[0]
    hp.text = ""
    r = hp.add_run(f"{srs.title} — {srs.product}")
    r.font.size, r.font.color.rgb = Pt(8.5), GREY
    hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT

    fp = sec.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for part in (f"Version {srs.version} · {srs.status.split(' —')[0]} · Page ",):
        run = fp.add_run(part)
        run.font.size, run.font.color.rgb = Pt(8.5), GREY
    _field(fp, "PAGE", "1")
    run = fp.add_run(" of ")
    run.font.size, run.font.color.rgb = Pt(8.5), GREY
    _field(fp, "NUMPAGES", "1")
    for run in fp.runs:
        run.font.size = Pt(8.5)


# ── Front matter ───────────────────────────────────────────────────────

def _front_heading(doc, text: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(text)
    r.bold, r.font.size, r.font.color.rgb = True, Pt(14), NAVY


def _title_page(doc, srs: SRSDocument) -> None:
    for _ in range(6):
        doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(srs.title)
    r.bold, r.font.size, r.font.color.rgb = True, Pt(26), NAVY

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("for")
    r.font.size, r.font.color.rgb = Pt(13), GREY

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(srs.product)
    r.bold, r.font.size = True, Pt(20)

    doc.add_paragraph()
    for label, value in (("Version", srs.version), ("Date", srs.date),
                         ("Status", srs.status), ("Prepared by", srs.prepared_by),
                         ("Conforms to", srs.standard)):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(2)
        a = p.add_run(f"{label}: ")
        a.bold, a.font.size = True, Pt(10.5)
        b = p.add_run(value)
        b.font.size = Pt(10.5)

    doc.add_paragraph()
    t = doc.add_table(rows=1, cols=1)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = t.rows[0].cells[0]
    _cell_text(cell, srs.notice, size=9.5, color=GREY)
    _shade(cell, "FFF4E5")
    _widths(t, [1], total=TEXT_W - 2)
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def _control_page(doc, srs: SRSDocument) -> None:
    _front_heading(doc, "Revision history")
    _table(doc, Table(["Version", "Date", "Description", "Author"], srs.revision_history,
                      widths=[1.2, 1.3, 4.5, 1.6]))
    _front_heading(doc, "Approvals")
    p = doc.add_paragraph("This specification becomes the requirements baseline only when "
                          "signed by each role below (problem statement §16: final "
                          "requirement baselines and regulatory interpretations require "
                          "human approval).")
    p.runs[0].font.size = Pt(10)
    _table(doc, Table(["Role", "Name", "Signature", "Date"], srs.approvals,
                      widths=[3, 2.5, 2.5, 1.5]), row_height=Cm(0.9))
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    _front_heading(doc, "Table of contents")
    p = doc.add_paragraph()
    _field(p, 'TOC \\o "1-3" \\h \\z \\u',
           "Right-click here and choose Update Field to build the table of contents.")


# ── Blocks ─────────────────────────────────────────────────────────────

# A table this short is kept on one page, with its heading: split, a use-case
# specification left its heading and header row stranded at a page foot.
KEEP_TOGETHER_ROWS = 15


def _keep_rows_together(table, upto: int) -> None:
    for row in table.rows[:upto]:
        _cant_split(row)
        for cell in row.cells:
            for par in cell.paragraphs:
                par.paragraph_format.keep_with_next = True


def _drop_trailing_spacer(doc) -> None:
    """Remove an empty spacer paragraph left after a table before a heading.

    Headings carry their own space above. A spacer that falls just past a page
    foot, followed by a Heading 1 with page-break-before, printed a blank page."""
    body = doc.element.body
    els = [e for e in body if e.tag != qn("w:sectPr")]
    if len(els) < 2 or els[-1].tag != qn("w:p") or els[-2].tag != qn("w:tbl"):
        return
    last = els[-1]
    if not "".join(last.itertext()).strip() and not last.findall(".//" + qn("w:br")):
        body.remove(last)


def _table(doc, t: Table, row_height=None) -> None:
    if t.caption:
        cp = doc.add_paragraph()
        cp.paragraph_format.keep_with_next = True
        r = cp.add_run(t.caption)
        r.italic, r.font.size = True, Pt(9)
    table = doc.add_table(rows=1, cols=len(t.headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    dense = len(t.headers) >= 5
    size = 8.5 if dense else 9.5
    for cell, h in zip(table.rows[0].cells, t.headers):
        _cell_text(cell, h, bold=True, size=size)
        _shade(cell, HEADER_FILL)
    _repeat_header(table.rows[0])
    for row in t.rows:
        cells = table.add_row().cells
        for cell, val in zip(cells, row):
            _cell_text(cell, str(val), size=size)
        if row_height is not None:
            table.rows[-1].height = row_height
    _widths(table, t.widths)
    # Short tables stay whole; long ones at least keep the header row with the
    # first data row. keep_with_next on every row but the last chains the table.
    _keep_rows_together(table, len(table.rows) - 1 if len(t.rows) <= KEEP_TOGETHER_ROWS else 1)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def _requirement(doc, req: Requirement) -> None:
    table = doc.add_table(rows=1, cols=2)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    top = table.rows[0].cells
    head = top[0].merge(top[1])
    head.text = ""
    p = head.paragraphs[0]
    p.paragraph_format.space_after = Pt(2)
    a = p.add_run(f"{req.id}   ")
    a.bold, a.font.size, a.font.color.rgb = True, Pt(10.5), NAVY
    b = p.add_run(req.statement)
    b.bold, b.font.size = True, Pt(10.5)
    _shade(head, REQ_FILL)
    _cant_split(table.rows[0])
    for label, value in req.attributes:
        cells = table.add_row().cells
        _cell_text(cells[0], label, bold=True, size=9)
        _shade(cells[0], LABEL_FILL)
        _cell_text(cells[1], value, size=9,
                   color=RGBColor(0xC0, 0x00, 0x00) if value.startswith(("TBD", "Names "))
                   or "unsupported" in value or "does NOT" in value else None)
        _cant_split(table.rows[-1])
    # Keep the whole block on one page where it fits.
    for row in table.rows[:-1]:
        for cell in row.cells:
            for par in cell.paragraphs:
                par.paragraph_format.keep_with_next = True
    _widths(table, [3.4, 13.2])
    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def _para(doc, b: Para) -> None:
    p = doc.add_paragraph()
    if b.lead:
        r = p.add_run(b.lead + " ")
        r.bold = True
    r = p.add_run(b.text)
    if b.style == "muted":
        r.italic, r.font.color.rgb, r.font.size = True, GREY, Pt(10)
    elif b.style == "note":
        r.font.color.rgb, r.font.size = RGBColor(0x9C, 0x57, 0x00), Pt(10)
    elif b.style == "quote":
        p.paragraph_format.left_indent = Cm(1)
        r.italic = True


def _bullets(doc, b: Bullets) -> None:
    for item in b.items:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_after = Pt(2)
        p.add_run(item)


# ── Entry point ────────────────────────────────────────────────────────

def render_docx(srs: SRSDocument, path) -> None:
    doc = Document()
    _setup(doc)
    _header_footer(doc, srs)
    cp = doc.core_properties
    cp.title = f"{srs.title} — {srs.product}"
    cp.subject = srs.standard
    cp.author = "Syntheia"
    cp.comments = srs.notice[:250]          # core properties cap at 255 characters

    _title_page(doc, srs)
    _control_page(doc, srs)

    for depth, sec in srs.walk():
        level = min(depth, 4)
        _drop_trailing_spacer(doc)
        doc.add_heading(sec.label(depth), level=level)
        for i, b in enumerate(sec.blocks):
            if isinstance(b, Requirement):
                _requirement(doc, b)
            elif isinstance(b, Table):
                _table(doc, b)
            elif isinstance(b, Bullets):
                _bullets(doc, b)
            elif isinstance(b, Para):
                _para(doc, b)
                # A line that introduces what follows stays on its page.
                nxt = sec.blocks[i + 1] if i + 1 < len(sec.blocks) else None
                if isinstance(nxt, (Table, Bullets, Requirement)):
                    doc.paragraphs[-1].paragraph_format.keep_with_next = True

    # Ask Word to refresh the TOC and page fields when the file is opened.
    upd = OxmlElement("w:updateFields")
    upd.set(qn("w:val"), "true")
    doc.settings.element.append(upd)
    doc.save(str(path))
