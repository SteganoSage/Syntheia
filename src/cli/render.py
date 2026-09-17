"""Rich rendering for the interactive application.

`src/report.py` stays as the plain-text renderer -- it is what redirects to a
file, what the eval loop reads, and what works on a dumb terminal. This module
is the same information laid out for a human reading it live.

The layout has one job beyond looking decent: **make uncertainty louder than
conclusions.** Every failure in this project's history came from confident
output that nobody checked -- a metric that read 100% on a field the pipeline
discarded, a tie presented as a recommendation, a fabricated citation with a
real-looking id. So an ungrounded claim is red, a tie is called a tie in the
headline, an overruled answer shows as overruled, and a derived column says it
is derived. A reader skimming this should find the soft spots first.
"""

from __future__ import annotations

import textwrap

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

# One place for the palette, so a change is a change here and not 40 edits.
C_OK = "green"
C_WARN = "yellow"
C_BAD = "red"
C_DIM = "dim"
C_KEY = "cyan"
C_HEAD = "bold bright_white"


def banner(provider: str, classifier: str, grounding: str, mode: str) -> None:
    t = Table.grid(padding=(0, 2))
    t.add_column(style=C_DIM, justify="right")
    t.add_column()
    t.add_row("model", provider)
    t.add_row("classifier", classifier)
    t.add_row("grounding", grounding)
    t.add_row("mode", f"[{C_KEY}]{mode}[/]")
    console.print(Panel(t, title="[bold]Syntheia[/]",
                        subtitle="[dim]/help for commands[/]",
                        border_style=C_KEY, padding=(1, 2)))


def echo_input(text: str, label: str = "scenario") -> None:
    """Confirm what was submitted.

    prompt_toolkit leaves the typed text on screen, which looks identical to
    text still being edited -- there is no visual moment of "sent". Reprinting
    it behind a rule gives the submit an unmistakable marker, and it stays in
    scrollback as the record of exactly what a given run was asked.
    """
    words = len(text.split())
    console.print()
    console.rule(f"[{C_KEY}]▶ {label} submitted[/] [dim]· {words} words[/]",
                 style=C_KEY, align="left")
    # Wrapped explicitly: letting rich wrap a Text puts the gutter on the first
    # line only, so continuation lines lose the bar and the quote stops reading
    # as a quote.
    width = max(30, console.width - 6)
    quoted = Text()
    for para in (text.splitlines() or [""]):
        for line in (textwrap.wrap(para, width) or [""]):
            quoted.append("  │ ", style=C_KEY)
            quoted.append(f"{line}\n", style=C_DIM)
    console.print(quoted, end="")


def requirement_panel(req, grounding: dict | None) -> Panel:
    g = grounding or {}
    body = Table.grid(padding=(0, 1))
    body.add_column(style=C_DIM, justify="right", width=14)
    body.add_column(overflow="fold")

    # The statement goes in the body, not the panel title: rich truncates a
    # title to the panel width, and a truncated requirement is unreadable.
    body.add_row("", Text(req.statement, style="bold"))
    body.add_row("", "")
    body.add_row("categories", ", ".join(req.category) or "[dim]none[/]")
    body.add_row("stakeholder", req.source_stakeholder or "[dim]—[/]")
    body.add_row("rationale", req.business_justification or "[dim]—[/]")
    body.add_row("acceptance", req.acceptance_criteria or "[dim]—[/]")

    if g.get("citations"):
        cites = Text()
        for c in g["citations"]:
            cites.append(f"{c['confidence']:.2f}  ", style=C_OK)
            cites.append(f"{c['citation']}\n")
        body.add_row("[green]grounded[/]", cites)
    elif g.get("needs_review") and req.applicable_regulations:
        # The model ASSERTED a regulation applies and the corpus cannot
        # corroborate it. This is §6's escalation and it must not read like a
        # footnote -- it is the most important line in the panel.
        body.add_row(
            f"[{C_BAD}]UNGROUNDED[/]",
            Text.from_markup(
                f"[{C_BAD}]claims {', '.join(req.applicable_regulations)} — "
                f"nothing in the knowledge base supports it "
                f"(best match {g['confidence']:.2f})[/]\n"
                f"[dim]needs human review[/]"),
        )
    elif g.get("needs_review"):
        # Claimed nothing, found nothing. Not every requirement is regulatory --
        # "the system shall be tested at every stage" is not, and colouring it
        # red would train the reader to ignore the colour that matters.
        body.add_row("[dim]regulations[/]",
                     f"[dim]none claimed, none found "
                     f"(best match {g['confidence']:.2f})[/]")
    elif req.applicable_regulations:
        body.add_row("regulations", ", ".join(req.applicable_regulations))

    if g.get("precedents"):
        # A tender is not a binding source. Kept visually distinct so the two
        # can never be read as the same kind of citation.
        prec = Text(no_wrap=False)
        for c in g["precedents"][:2]:
            prec.append(f"{c['confidence']:.2f}  ", style=C_DIM)
            prec.append(f"{c['citation']}\n", style=C_DIM)
        body.add_row("[dim]precedent[/]", prec)

    risk_style = {"High": C_BAD, "Medium": C_WARN}.get(req.risk_level, C_DIM)
    body.add_row("risk / conf",
                 Text.from_markup(f"[{risk_style}]{req.risk_level}[/]"
                                  f" [dim]·[/] {req.confidence_score}"))

    prio = {"High": C_BAD, "Medium": C_WARN}.get(req.priority, C_DIM)
    return Panel(
        body,
        title=f"[{C_HEAD}]{req.id}[/]  [{prio}]{req.priority} priority[/]",
        title_align="left",
        border_style=C_DIM, padding=(0, 1),
    )


def print_requirements(requirements, grounding, elapsed=None) -> None:
    n = len(requirements)
    timing = f" [dim]in {elapsed:.0f}s[/]" if elapsed is not None else ""
    console.print(f"\n[{C_HEAD}]{n} requirement{'s' if n != 1 else ''}[/]{timing}\n")
    for req in requirements:
        console.print(requirement_panel(req, (grounding or {}).get(req.id)))


def print_quality(report) -> None:
    if not report:
        return
    passed = sum(1 for r in report.result_list if r.passed)
    flagged = len(report.result_list) - passed

    rows = []
    for qr in report.result_list:
        if qr.passed:
            continue
        for issue in qr.issues:
            sev = str(issue.severity)
            style = {"critical": C_BAD, "high": C_BAD,
                     "medium": C_WARN}.get(sev.lower(), C_DIM)
            rows.append((qr.requirement_id, sev, style,
                         issue.issue_type, issue.description, issue.suggestion))

    if not rows:
        console.print(Panel(f"[{C_OK}]all {passed} requirements passed[/]",
                            title="[bold]quality[/]", border_style=C_OK,
                            padding=(0, 1)))
        return

    t = Table(box=None, padding=(0, 1), show_header=True, header_style=C_DIM)
    t.add_column("req", width=9)
    t.add_column("severity", width=9)
    t.add_column("issue", overflow="fold")
    for rid, sev, style, kind, desc, fix in rows:
        t.add_row(rid, f"[{style}]{sev}[/]",
                  Text.from_markup(f"[{style}]{kind}[/] — {desc}\n"
                                   f"[dim]→ {fix}[/]"))
    console.print(Panel(
        t, title=f"[bold]quality[/]  [green]{passed} passed[/] "
                 f"[{C_WARN}]{flagged} flagged[/]",
        border_style=C_WARN, padding=(0, 1)))


def print_sdlc(rec) -> None:
    if not rec:
        return

    top = rec.recommendation
    derived = any(m.is_derived and m.name == top for m in rec.ranking)

    head = Text()
    if rec.tied_with:
        # A tie is not a recommendation. Two of five runs were decided
        # alphabetically and printed as findings before this existed.
        head.append("TIED  ", style=f"bold {C_WARN}")
        head.append(" = ".join(rec.tied_with), style=f"bold {C_WARN}")
        head.append("\nthe framework does not separate these. Listed first by "
                    "name, which is not a preference.", style=C_DIM)
    else:
        head.append(top, style=f"bold {C_OK}")
        if derived:
            head.append("   (derived column — not in the supplied framework)",
                        style=C_WARN)

    scores = Table(box=None, padding=(0, 1), show_header=False)
    scores.add_column(width=2)
    scores.add_column(width=13)
    scores.add_column(justify="right", width=5)
    scores.add_column(style=C_DIM)
    for s in rec.ranking:
        is_top = s.name == top or s.name in (rec.tied_with or [])
        bar = "█" * round(s.score * 18)
        scores.add_row(
            "▸" if is_top else "",
            f"[bold]{s.name}[/]" if is_top else f"[dim]{s.name}[/]",
            f"{s.score:.0%}",
            Text.from_markup(
                f"[{C_OK if is_top else C_DIM}]{bar}[/] "
                f"{len(s.matched)}/{len(s.matched) + len(s.missed)}"
                + ("  derived" if s.is_derived else "")),
        )

    answers = Table(box=None, padding=(0, 1), show_header=False)
    answers.add_column(width=7, style=C_KEY)
    answers.add_column(width=4)
    answers.add_column(overflow="fold", style=C_DIM)
    for a in rec.criteria:
        overruled = a.reasoning.startswith("UNSUPPORTED")
        label, style = {"yes": ("yes", C_OK), "no": ("no", C_WARN),
                        "not_stated": ("—", C_DIM)}[a.answer]
        if overruled:
            ev = f"[{C_BAD}]dropped: nothing traceable to the scenario[/]"
        else:
            ev = ", ".join(e[:52] for e in a.evidence) or "[dim]nothing cited[/]"
        answers.add_row(a.question_id, f"[{style}]{label}[/]", ev)

    cov = Table(box=None, padding=(0, 1), show_header=False)
    cov.add_column(width=22, style=C_DIM)
    cov.add_column()
    for c in rec.coverage.values():
        ok = c.answered == c.total
        style = C_OK if ok else (C_WARN if c.assessed else C_BAD)
        cov.add_row(c.name, f"[{style}]{c.answered}/{c.total}[/] "
                            f"[dim]{c.source}[/]")

    blocks = [head, "", Text("framework scoring", style=C_DIM), scores,
              "", Text("inferred answers", style=C_DIM), answers,
              "", Text("coverage", style=C_DIM), cov]

    if rec.cross_check_ranking:
        agrees = rec.cross_check_agrees
        blocks += ["", Text.from_markup(
            f"[{C_OK if agrees else C_WARN}]cross-check "
            f"{'agrees' if agrees else 'disagrees'}[/][dim] — ranked without "
            f"seeing scores: {' > '.join(rec.cross_check_ranking[:4])}[/]")]

    console.print(Panel(Group(*blocks), title="[bold]SDLC recommendation[/]",
                        border_style=C_WARN if rec.tied_with else C_OK,
                        padding=(1, 2)))

    for j in rec.justifications:
        console.print(Panel(j.justification, title=f"[dim]{j.model}[/]",
                            title_align="left", border_style=C_DIM,
                            padding=(0, 1)))
    if rec.summary:
        console.print(Panel(rec.summary, border_style=C_DIM, padding=(0, 1)))


def print_review(reasons: list[str]) -> None:
    if not reasons:
        console.print(Panel(f"[{C_OK}]nothing flagged for review[/]",
                            border_style=C_OK, padding=(0, 1)))
        return
    body = Text()
    for r in reasons:
        body.append("• ", style=C_BAD)
        body.append(f"{r}\n")
    console.print(Panel(body, title=f"[bold {C_BAD}]needs review[/]",
                        border_style=C_BAD, padding=(0, 1)))


def print_run(state: dict, elapsed: float | None = None) -> None:
    print_requirements(state.get("requirements", []),
                       state.get("grounding", {}), elapsed)
    print_quality(state.get("quality_report"))
    print_sdlc(state.get("sdlc"))
