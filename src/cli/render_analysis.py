"""Terminal rendering for Phase B: conflicts, compliance, security, risk, questions.

Same rule as `render.py`: make uncertainty louder than conclusions. A possible
conflict says "possible", a rule nobody has verified says so, a practice is
not dressed as a regulation, and an unprotected data class is red.
"""

from __future__ import annotations

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.cli.render import C_BAD, C_DIM, C_KEY, C_OK, C_WARN, console

_SEV = {"Critical": C_BAD, "High": C_BAD, "Medium": C_WARN, "Low": C_DIM}
SHOW = 8                      # rows per section before "and N more"


def _more(n: int) -> str:
    return f"[dim]… and {n} more (in --json)[/]" if n > 0 else ""


def print_conflicts(conflicts, requirements) -> None:
    if conflicts is None:
        return
    if not conflicts:
        console.print(Panel(f"[{C_OK}]no contradictions or duplicates found[/]",
                            title="[bold]consistency[/]", border_style=C_OK, padding=(0, 1)))
        return
    by_id = {r.id: r for r in requirements}
    body = Text()
    for c in conflicts[:SHOW]:
        tag = ("" if c.confirmed else "possible ") + c.kind
        body.append(f"{tag.upper():20s}", style=_SEV[c.severity] if c.kind == "conflict" else C_DIM)
        body.append(" ".join(c.requirement_ids) + "\n", style=C_KEY)
        for rid in c.requirement_ids[:2]:
            r = by_id.get(rid)
            if r:
                body.append(f"    {rid}: {r.statement}\n", style=C_DIM)
        body.append(f"    {c.explanation}\n")
        if c.resolution:
            body.append(f"    → {c.resolution}\n", style=C_DIM)
    if len(conflicts) > SHOW:
        body.append_text(Text.from_markup(_more(len(conflicts) - SHOW)))
    n_conf = sum(1 for c in conflicts if c.kind == "conflict")
    console.print(Panel(body, title=f"[bold]consistency[/]  [{C_BAD}]{n_conf} conflict[/] "
                                    f"[dim]{len(conflicts) - n_conf} duplicate[/]",
                        border_style=C_WARN, padding=(0, 1)))


def _finding_table(findings) -> Table:
    t = Table(box=None, padding=(0, 1), show_header=True, header_style=C_DIM)
    t.add_column("rule", width=15)
    t.add_column("sev", width=8)
    t.add_column("finding", overflow="fold")
    for f in findings[:SHOW]:
        cite = f.source.citation if f.source else (f.reference or "")
        basis = "" if f.basis == "regulation" else "  [dim](practice, not regulation)[/]"
        status = "[yellow]weak[/] — " if f.status == "weak" else ""
        line = f"{status}{f.title}{basis}\n"
        if f.detail:
            line += f"[dim]{f.detail}[/]\n"
        if cite:
            line += f"[dim]{cite[:110]}[/]\n"
        if f.proposed_requirement:
            line += f"[dim]→ {f.proposed_requirement}[/]"
        if f.requires_legal_review:
            line += f"\n[{C_WARN}]needs compliance/legal sign-off[/]"
        t.add_row(f.rule_id, f"[{_SEV[f.severity]}]{f.severity}[/]",
                  Text.from_markup(line.rstrip()))
    return t


def print_compliance(report) -> None:
    if report is None:
        return
    gaps = report.gaps
    met = [f for f in report.rule_findings if f.status == "satisfied"]
    if gaps:
        body = _finding_table(gaps)
        title = (f"[bold]compliance[/]  [{C_BAD}]{len(gaps)} gap[/] "
                 f"[{C_OK}]{len(met)} covered[/]")
        console.print(Panel(body, title=title, border_style=C_BAD, padding=(0, 1)))
        if len(gaps) > SHOW:
            console.print(_more(len(gaps) - SHOW))
    elif report.rule_findings:
        console.print(Panel(f"[{C_OK}]all {len(met)} applicable obligations are covered[/]",
                            title="[bold]compliance[/]", border_style=C_OK, padding=(0, 1)))

    if report.mappings:
        t = Table(box=None, padding=(0, 1), show_header=True, header_style=C_DIM)
        t.add_column("req", width=8)
        t.add_column("coverage", width=14)
        t.add_column("obligation", overflow="fold")
        style = {"addressed": C_OK, "partial": C_WARN, "not_addressed": C_BAD,
                 "not_applicable": C_DIM, "unclear": C_WARN}
        for m in report.mappings[:SHOW]:
            text = (m.obligation or "[dim](obligation not traceable to the clause)[/]")
            text += f"\n[dim]{m.source.citation[:100]}[/]"
            if m.source.jurisdiction:
                text += f" [dim]· {m.source.jurisdiction}[/]"
            if m.candidates_checked > 1:
                text += f" [dim]· {m.candidates_checked} clauses read[/]"
            if m.missing:
                text += f"\n[dim]→ {m.missing}[/]"
            if m.note:
                text += f"\n[{C_WARN}]{m.note}[/]"
            t.add_row(m.requirement_id, f"[{style[m.coverage]}]{m.coverage}[/]",
                      Text.from_markup(text))
        console.print(Panel(t, title="[bold]clause mapping[/]", border_style=C_KEY,
                            padding=(0, 1)))

    if report.obligations:
        body = Text()
        for o in report.obligations[:SHOW]:
            body.append(f"{o.kind:12s}", style=C_KEY)
            body.append(f"{o.text}\n")
            if o.citation:
                body.append(f"{'':12s}{o.citation[:100]}\n", style=C_DIM)
        console.print(Panel(body, title="[bold]retention, reporting and approval obligations[/]",
                            border_style=C_DIM, padding=(0, 1)))


def print_security(report) -> None:
    if report is None:
        return
    if report.data_inventory:
        t = Table(box=None, padding=(0, 1), show_header=True, header_style=C_DIM)
        t.add_column("data", width=24)
        t.add_column("found in", overflow="fold")
        t.add_column("protected by", overflow="fold")
        for d in report.data_inventory:
            prot = (", ".join(d.protected_by) if d.protected
                    else f"[{C_BAD if d.sensitivity == 'high' else C_WARN}]nothing[/]")
            t.add_row(d.label, ", ".join(d.found_in[:5]), Text.from_markup(prot))
        console.print(Panel(t, title="[bold]sensitive data[/]", border_style=C_KEY,
                            padding=(0, 1)))
    gaps = report.gaps
    if gaps:
        console.print(Panel(_finding_table(gaps),
                            title=f"[bold]security controls[/]  [{C_BAD}]{len(gaps)} missing[/]",
                            border_style=C_WARN, padding=(0, 1)))


def print_risks(risks) -> None:
    if not risks:
        return
    t = Table(box=None, padding=(0, 1), show_header=True, header_style=C_DIM)
    t.add_column("id", width=9)
    t.add_column("L×I", width=6)
    t.add_column("risk", overflow="fold")
    t.add_column("owner", width=18)
    for r in risks[:SHOW + 2]:
        t.add_row(r.id, f"[{_SEV[r.rating]}]{r.likelihood}×{r.impact}={r.score}[/]",
                  Text.from_markup(f"{r.title} [dim]({r.category})[/]\n[dim]{r.cause}[/]"),
                  r.owner)
    console.print(Panel(t, title=f"[bold]risk register[/]  {len(risks)} risks",
                        border_style=C_KEY, padding=(0, 1)))
    if len(risks) > SHOW + 2:
        console.print(_more(len(risks) - SHOW - 2))


def print_clarifications(result) -> None:
    if result is None:
        return
    if result.changes:
        body = Text()
        for ch in result.changes:
            body.append(f"{ch.requirement_id} {ch.field}\n", style=C_KEY)
            if ch.before:
                body.append(f"   was: {ch.before}\n", style=C_DIM)
            body.append(f"   now: {ch.after}\n")
            body.append(f"   because you said: “{ch.reason}”\n", style=C_DIM)
        console.print(Panel(body, title="[bold]changed after your answer[/]",
                            border_style=C_OK, padding=(0, 1)))
    open_qs = result.open
    if not open_qs:
        return
    body = Text()
    for q in open_qs[:SHOW]:
        body.append(f"{q.id} ", style=C_KEY)
        body.append(f"{q.question}\n")
        body.append(f"       {q.why}\n", style=C_DIM)
    if len(open_qs) > SHOW:
        body.append_text(Text.from_markup(_more(len(open_qs) - SHOW)))
    console.print(Panel(body, title=f"[bold]open questions[/]  {len(open_qs)}",
                        border_style=C_WARN, padding=(0, 1)))


def print_dropped(dropped) -> None:
    """Statements extraction produced that were facts about the project, not the system."""
    if not dropped:
        return
    body = Text()
    for d in dropped:
        body.append(f"{d['id']} ", style=C_DIM)
        body.append(f"{d['statement']}\n")
        body.append(f"    {d['reason']} — a fact about the project, not the system; "
                    f"the SDLC engine reads it from the scenario\n", style=C_DIM)
    console.print(Panel(body, title="[bold]removed: not system requirements[/]",
                        border_style=C_DIM, padding=(0, 1)))


def print_analysis(state: dict) -> None:
    print_dropped(state.get("dropped"))
    if not state.get("run_analysis", True):
        return
    reqs = state.get("requirements") or []
    print_conflicts(state.get("conflicts"), reqs)
    print_compliance(state.get("compliance"))
    print_security(state.get("security"))
    print_risks(state.get("risk_register"))
    print_clarifications(state.get("clarifications"))
