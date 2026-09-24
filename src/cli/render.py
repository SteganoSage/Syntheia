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

import time
from contextlib import contextmanager

from rich.console import Console, Group
from rich.live import Live
from rich.spinner import Spinner
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


# ── One terminal, one owner ────────────────────────────────────────────
#
# Rich's Live repaints on its own thread, so anything else writing to the same
# terminal loses: the pipeline checklist drew straight over the interview
# prompt, and starting the interview's own spinner made two Live displays fight
# for the screen while prompt_toolkit tried to render an input line into the
# middle of it.
#
# The interview happens deep inside the graph -- sdlc node -> recommend_sdlc ->
# ask_user -- so it cannot reach the Live that scenario mode started. Hence a
# module-level handle: whoever owns the screen registers here, and anyone that
# needs it quiet borrows it through `suspended()`.

_active_live = None
_active_owner = None
_muted = False


def is_muted() -> bool:
    """True while something else owns the screen and progress must draw nothing.

    Belt and braces on top of stopping the Live. Stopping alone proved fragile:
    the checklist still appeared in the middle of the interview, because there
    are several paths that can start a Live and only one that knows the
    interview is happening. A renderable that returns nothing cannot draw over
    anything, whoever restarts it.
    """
    return _muted


def set_live(live, owner=None) -> None:
    """Register the Live that currently owns the terminal (None to clear).

    `owner` is the object being rendered -- the progress checklist. It has to be
    passed explicitly: rich wraps the renderable in a Group once the Live
    starts, so `live.renderable` afterwards is NOT the object handed in, and
    reaching for `snapshot()` on it silently finds nothing.
    """
    global _active_live, _active_owner
    _active_live = live
    _active_owner = owner if live is not None else None


@contextmanager
def suspended(commit: bool = False):
    """Stop the active Live for the duration, then bring it back.

    Used around anything that reads input or starts a display of its own.
    Restarting is best-effort: a Live that has already been stopped by its owner
    must not resurrect itself here and start painting over whatever came next.

    `commit=True` is for a suspension the user will write underneath -- the
    interview. A transient Live erases itself on stop and redraws itself whole
    on start, so without this the finished checklist vanished, the questions and
    answers scrolled past, and then the entire checklist reappeared below them
    as if the run had restarted. Committing prints the progress permanently
    first and switches it to compact, so what comes back afterwards is only the
    one stage still working.
    """
    global _muted
    live = _active_live
    running = bool(live and getattr(live, "is_started", False))
    renderable = _active_owner if running else None
    was_muted, _muted = _muted, True

    if running:
        try:
            live.stop()
            if commit:
                # Freeze what has finished into the scrollback before anyone
                # writes below it.
                snap = getattr(renderable, "snapshot", None)
                if callable(snap):
                    console.print(snap())
                if hasattr(renderable, "compact"):
                    renderable.compact = True
        except Exception:
            running = False
    try:
        yield
    finally:
        _muted = was_muted
        if running:
            try:
                live.start()
            except Exception:
                pass


class Thinking:
    """A spinner whose caption rotates while a blocking call runs.

    Lazy for the same reason the pipeline checklist is: whatever it is waiting
    on holds the thread, so nothing calls `update()`. Rich re-renders an object
    with `__rich__` on its own refresh thread, which is the only way the caption
    moves and the clock ticks during a call that blocks for a minute.

    Used wherever the app would otherwise sit silent. Silence during a long call
    is indistinguishable from a hang -- which is how a runaway generation went
    undiagnosed for two sittings.
    """

    SWAP_SECONDS = 2.2

    def __init__(self, lines, style: str = C_KEY):
        self.lines = list(lines) or ["working"]
        self.style = style
        self._i = 0
        self._t0 = 0.0
        self._last = 0.0
        self._live = None

    def __rich__(self):
        now = time.perf_counter()
        if now - self._last > self.SWAP_SECONDS:
            self._i = (self._i + 1) % len(self.lines)
            self._last = now
        t = Table.grid(padding=(0, 1))
        t.add_column(width=2)
        t.add_column()
        t.add_row(Spinner("dots", style=self.style),
                  Text.from_markup(f"[dim italic]{self.lines[self._i]}[/]  "
                                   f"[dim]{now - self._t0:.0f}s[/]"))
        return t

    def __enter__(self):
        self._t0 = self._last = time.perf_counter()
        # Take the screen from whoever has it: rich refuses two simultaneous
        # Live displays, and the failure mode is a corrupted terminal rather
        # than an error.
        self._borrow = suspended()
        self._borrow.__enter__()
        self._live = Live(self, console=console, refresh_per_second=8,
                          transient=True)
        self._live.start()
        return self

    def __exit__(self, *exc):
        try:
            self._live.stop()
        except Exception:
            pass
        try:
            self._borrow.__exit__(None, None, None)
        except Exception:
            pass
        return False


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


def print_sdlc(rec, detail: bool = False) -> None:
    """The recommendation, as an argument rather than a scorecard.

    The percentages, the match counts and the coverage fractions are gone from
    the default view. They are how the decision is computed, not what the
    decision IS, and leading with "8/9, 89%" invites a reader to audit the
    instrument instead of the reasoning -- while making a partial assessment
    look like a measurement. The marked answer this output is modelled on has no
    numbers in it either: it names the model, says which characteristics of the
    project favour it, what that buys, and what it costs.

    Nothing is lost. `--detail` prints the full working, and the JSON always
    carries every answer, score and coverage figure.
    """
    if not rec:
        return

    top = rec.recommendation
    derived = any(m.is_derived and m.name == top for m in rec.ranking)

    head = Text()
    if rec.tied_with:
        head.append("TIED  ", style=f"bold {C_WARN}")
        head.append(" and ".join(rec.tied_with), style=f"bold {C_WARN}")
        head.append("\nthe evidence does not separate these two. More about "
                    "the project would.", style=C_DIM)
    else:
        head.append(top, style=f"bold {C_OK}")
        if derived:
            head.append("   (our reading — not one of the six supplied models)",
                        style=C_WARN)

    others = [m.name for m in rec.ranking[1:4]
              if m.name not in (rec.tied_with or [])]
    blocks = [head]
    if others:
        blocks += ["", Text.from_markup(
            f"[dim]also considered: {', '.join(others)}[/]")]

    if rec.cross_check_ranking:
        agrees = rec.cross_check_agrees
        blocks += ["", Text.from_markup(
            f"[{C_OK if agrees else C_WARN}]a second, independent read "
            f"{'agreed' if agrees else 'disagreed'}[/]"
            + ("" if agrees else
               f"[dim] — it would have chosen "
               f"{rec.cross_check_ranking[0]}[/]"))]

    console.print(Panel(Group(*blocks), title="[bold]recommended approach[/]",
                        border_style=C_WARN if rec.tied_with else C_OK,
                        padding=(1, 2)))

    for j in rec.justifications:
        body = Table.grid(padding=(0, 1))
        body.add_column(style=C_DIM, justify="right", width=11)
        body.add_column(overflow="fold")

        fits = Text()
        for b in j.why_it_fits:
            fits.append("• ", style=C_OK)
            fits.append(f"{b}\n")
        body.add_row("[green]fits[/]", fits or Text("—", style=C_DIM))

        if j.advantages:
            adv = Text()
            for a in j.advantages:
                adv.append(f"{a.advantage}\n", style="bold")
                adv.append(f"  {a.relevance}\n", style=C_DIM)
            body.add_row("buys you", adv)

        if j.gives_up:
            body.add_row(f"[{C_WARN}]gives up[/]", Text(j.gives_up, style=C_WARN))

        console.print(Panel(body, title=f"[bold]{j.model}[/]",
                            title_align="left", border_style=C_DIM,
                            padding=(0, 1)))

    if rec.summary:
        console.print(Panel(rec.summary, border_style=C_DIM, padding=(0, 1)))
    if rec.shortcut:
        console.print(Panel(Text(rec.shortcut, style=f"bold {C_KEY}"),
                            title="[dim]in one line[/]", title_align="left",
                            border_style=C_KEY, padding=(0, 1)))

    if detail:
        print_sdlc_detail(rec)


def print_sdlc_detail(rec) -> None:
    """The full working: every score, every answer, every coverage figure.

    Kept for us, not for the report. This is what makes a disagreement
    debuggable, and it is exactly what should not be the first thing a reader
    sees.
    """
    scores = Table(box=None, padding=(0, 1), show_header=False)
    scores.add_column(width=2); scores.add_column(width=13)
    scores.add_column(justify="right", width=5); scores.add_column(style=C_DIM)
    for s in rec.ranking:
        is_top = s.name == rec.recommendation or s.name in (rec.tied_with or [])
        scores.add_row("▸" if is_top else "",
                       f"[bold]{s.name}[/]" if is_top else f"[dim]{s.name}[/]",
                       f"{s.score:.0%}",
                       f"{len(s.matched)}/{len(s.matched) + len(s.missed)}"
                       + ("  derived" if s.is_derived else ""))

    answers = Table(box=None, padding=(0, 1), show_header=False)
    answers.add_column(width=4); answers.add_column(overflow="fold")
    for a in rec.criteria:
        label, style = {"yes": ("yes", C_OK), "no": ("no", C_WARN),
                        "not_stated": ("—", C_DIM)}[a.answer]
        answers.add_row(f"[{style}]{label}[/]", f"[dim]{a.reasoning[:88]}[/]")

    cov = Table(box=None, padding=(0, 1), show_header=False)
    cov.add_column(width=22, style=C_DIM); cov.add_column()
    for c in rec.coverage.values():
        cov.add_row(c.name, f"{c.answered}/{c.total} [dim]{c.source}[/]")

    console.print(Panel(
        Group(Text("scoring", style=C_DIM), scores, "",
              Text("answers", style=C_DIM), answers, "",
              Text("coverage", style=C_DIM), cov),
        title="[dim]working (--detail)[/]", border_style=C_DIM, padding=(1, 2)))


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


def print_run(state: dict, elapsed: float | None = None,
              detail: bool = False) -> None:
    print_requirements(state.get("requirements", []),
                       state.get("grounding", {}), elapsed)
    print_quality(state.get("quality_report"))
    print_sdlc(state.get("sdlc"), detail=detail)
