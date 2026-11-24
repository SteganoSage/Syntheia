"""Scenario mode: text in, grounded requirements and an SDLC recommendation out.

Two things here are interaction design rather than plumbing.

**Results print as they finish, not at the end.** The graph is six nodes and the
last one is the slowest -- the SDLC engine makes several LLM calls after
everything a reader actually wants is already computed. Batching the output
until the graph returns means staring at a spinner for minutes with finished
work sitting in memory. So each milestone prints the moment it is reached and
the spinner resumes underneath for whatever is still running.

**The spinner says what is happening, and it is true.** Generic filler
("Thinking...") tells a reader nothing and erodes trust the moment it is
obviously wrong. The lines rotate within the current stage, so they are always
an accurate description of the work in flight, and an elapsed counter ticks
regardless -- that counter is the real anti-"is-it-stuck" signal, because it
keeps moving even if a stage takes far longer than expected.
"""

from __future__ import annotations

import itertools
import time

from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from src.cli import render
from src.cli.render import C_DIM, C_KEY, C_OK, console

# Per stage: the label, how to summarise what it produced, and the chatter.
#
# The chatter is there to make a four-minute wait feel like something is alive.
# It is scoped to the running stage, so it is never wildly wrong about what is
# happening -- but it is not a progress log either, and it does not pretend to
# be. A little personality costs nothing and the alternative is a spinner and
# silence, which is what a hang looks like.
STAGES: list[dict] = [
    dict(node="retrieval", label="retrieving source material",
         summary=lambda s: _passages(s),
         chatter=["speed-reading the RBI",
                  "flipping through 800 pages of master direction",
                  "the regulator has thoughts about this",
                  "wading through legalese",
                  "judging documents by their section numbers",
                  "somewhere in here is a relevant clause",
                  "hoping someone wrote this down",
                  "consulting thirteen very serious PDFs",
                  "skipping the table of contents, obviously",
                  "asking the corpus nicely"]),
    dict(node="extraction", label="extracting requirements",
         summary=lambda s: f"{len(s.get('requirements') or [])} requirements",
         chatter=["turning vibes into 'the system shall'",
                  "pretending to be a business analyst",
                  "nobody says 'shall' in real life",
                  "inventing a stakeholder who cares",
                  "writing acceptance criteria nobody will read",
                  "resisting the urge to add a dashboard",
                  "deciding this counts as one requirement",
                  "your scenario was three sentences, give me a moment",
                  "no, team size is not a requirement",
                  "being very normal about the word 'shall'",
                  "extracting, allegedly",
                  "this is the fun part"]),
    dict(node="grounding", label="grounding against the knowledge base",
         summary=lambda s: _grounding(s),
         chatter=["demanding receipts",
                  "citation needed",
                  "asking 'says who?'",
                  "making sure we did not make that up",
                  "matching claims to clauses, suspiciously",
                  "the corpus has never heard of this",
                  "verifying, reluctantly",
                  "a tender is not a regulation, sorry",
                  "double-checking the paperwork"]),
    dict(node="classification", label="classifying",
         summary=lambda s: _categories(s),
         chatter=["RoBERTa has opinions",
                  "twenty-one milliseconds of pure judgement",
                  "is this Security or Privacy? yes",
                  "slapping labels on things",
                  "one label was never going to be enough",
                  "the classifier is suspiciously confident",
                  "thirteen categories, no mercy"]),
    dict(node="quality", label="checking quality",
         summary=lambda s: _quality(s),
         chatter=["being deliberately annoying",
                  "asking what 'user-friendly' means, again",
                  "nitpicking professionally",
                  "someone has to be the pedant",
                  "'real-time' is not a number",
                  "hunting weasel words",
                  "this is the part everyone hates",
                  "adjectives are not requirements",
                  "yes it is testable, prove it",
                  "no notes. actually, several notes"]),
    dict(node="sdlc", label="scoring the SDLC framework",
         summary=lambda s: (s["sdlc"].recommendation if s.get("sdlc") else "skipped"),
         chatter=["consulting the sacred tables",
                  "Waterfall versus V-Shaped, round forty-seven",
                  "agile? in this economy?",
                  "seven models enter, one leaves",
                  "counting on fingers",
                  "arguing with itself about integration",
                  "trying very hard not to just say Agile",
                  "ranking without peeking at the scores",
                  "doing arithmetic, slowly, out loud",
                  "the framework has spoken",
                  "asking whether anything could still change its mind",
                  "dropping the answers it cannot back up",
                  "writing this up like an exam answer",
                  "putting four months of project into one line",
                  "nearly there, promise"]),
]

# Where to interrupt the spinner and print. Requirements are only complete once
# classification has filled `category`, so that is the earliest honest point --
# printing them after extraction would show every one with an empty category and
# then need reprinting.
BY_NODE = {st["node"]: st for st in STAGES}

MILESTONES = {
    "classification": "requirements",
    "quality": "quality",
    "clarification": "analysis",
    "sdlc": "sdlc",
}


# Phase B stages, inserted between quality and sdlc. Kept as one block so the
# insertion point is the only line of the existing table that changed.
def _analysis_stages() -> list[dict]:
    def n(key):
        return lambda s: len(s.get(key) or [])

    def comp(s):
        c = s.get("compliance")
        return f"{len(c.gaps)} gaps, {len(c.mappings)} clauses mapped" if c else "skipped"

    def sec(s):
        c = s.get("security")
        if not c:
            return "skipped"
        bare = sum(1 for d in c.data_inventory if not d.protected)
        return f"{len(c.gaps)} controls missing, {bare} data classes unprotected"

    def clar(s):
        c = s.get("clarifications")
        if not c:
            return "skipped"
        return (f"{len(c.open)} open" + (f", {len(c.changes)} changes" if c.changes else ""))

    return [
        dict(node="conflicts", label="checking requirements against each other",
             summary=lambda s: f"{n('conflicts')(s)} found",
             chatter=["comparing every requirement with every other one",
                      "looking for two things that cannot both be true",
                      "spotting the same requirement in a different hat",
                      "asking the judge, one pair at a time"]),
        dict(node="compliance", label="mapping obligations",
             summary=comp,
             chatter=["reading the clause, not just the title",
                      "checking what the regulator actually requires",
                      "finding obligations nobody wrote down",
                      "five years means five years"]),
        dict(node="security", label="checking security & privacy controls",
             summary=sec,
             chatter=["counting the Aadhaar numbers",
                      "who can see what, exactly",
                      "is that encrypted? it should be encrypted"]),
        dict(node="risk", label="building the risk register",
             summary=lambda s: f"{n('risk_register')(s)} risks",
             chatter=["likelihood times impact, carry the one",
                      "imagining everything that could go wrong",
                      "assigning owners who will love this"]),
        dict(node="clarification", label="deciding what to ask",
             summary=clar,
             chatter=["turning findings into questions",
                      "contradictions first, always",
                      "writing it so a human can answer it"]),
    ]


_e = next(i for i, st in enumerate(STAGES) if st["node"] == "extraction")
STAGES.insert(_e + 1, dict(
    node="elaboration", label="adding implied requirements",
    summary=lambda s: f"{sum(1 for r in s.get('requirements') or [] if r.origin == 'implied')} implied",
    chatter=["thinking like a business analyst in the first workshop",
             "what happens when NSDL is down?",
             "someone will want to track their application",
             "not repeating what you already said"]))
_q = next(i for i, st in enumerate(STAGES) if st["node"] == "quality")
STAGES[_q + 1:_q + 1] = _analysis_stages()
BY_NODE = {st["node"]: st for st in STAGES}


def _passages(s) -> str:
    hits = s.get("context") or []
    if not hits:
        # Expected, not broken: an 8-word brief legitimately retrieves nothing
        # above MIN_CONTEXT_RELEVANCE, and per-requirement grounding still runs.
        return "none above the relevance floor"
    return f"{len(hits)} passages, top {hits[0]['score']:.2f}"


def _grounding(s) -> str:
    g = s.get("grounding") or {}
    reqs = {r.id: r.applicable_regulations for r in s.get("requirements") or []}
    cited = sum(1 for v in g.values() if v.get("citations"))
    unsupported = sum(1 for rid, v in g.items()
                      if v.get("needs_review") and reqs.get(rid))
    out = f"{cited} cited"
    if unsupported:
        out += f", {unsupported} unsupported"
    return out


def _categories(s) -> str:
    reqs = s.get("requirements") or []
    n = sum(len(r.category or []) for r in reqs)
    return f"{n} labels over {len(reqs)}" if reqs else "—"


def _quality(s) -> str:
    r = s.get("quality_report")
    if not r:
        return "skipped"
    passed = sum(1 for x in r.result_list if x.passed)
    return f"{passed}/{len(r.result_list)} passed"


class _Progress:
    """The live checklist, rendered lazily.

    Must be lazy. While a node runs, the `for chunk in stream()` loop is blocked
    -- no iteration happens, so nothing calls `live.update()`. A pre-built
    renderable would freeze the elapsed counter and the chatter for the whole of
    a six-minute SDLC call, which is precisely the wait this display exists to
    make bearable. Rich calls `__rich__` on every refresh, so putting the clock
    inside it means the counter ticks and the chatter rotates on Live's own
    thread, independent of the graph.
    """

    SWAP_SECONDS = 2.5          # Live refreshes ~8x/sec; text at that rate is noise

    def __init__(self, order: list[str], skipped: set[str]):
        self.order = order
        self.skipped = skipped
        # Set once the finished stages have been printed permanently -- after
        # that this renders only the stage still working, so the checklist is
        # not repeated below whatever was written in between.
        self.compact = False
        # Stages whose finished line has already been printed permanently. The
        # checklist is not one block that redraws -- it grows downward, a line
        # at a time, with results printed in between. Without this every
        # milestone reprinted the whole list under the panel it had just shown.
        self._committed: set[str] = set()
        self.done: dict[str, str] = {}
        self.t0 = time.perf_counter()
        self._cycle = None
        self._chatter = ""
        self._last_swap = 0.0
        self._current: str | None = None
        self.advance()

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def advance(self) -> None:
        """Point at the first unfinished stage and reset its chatter."""
        nxt = next((n for n in self.order if n not in self.done), None)
        if nxt != self._current:
            self._current = nxt
            if nxt:
                self._cycle = itertools.cycle(BY_NODE[nxt]["chatter"])
                self._chatter = next(self._cycle)
                self._last_swap = time.perf_counter()

    def finish(self, node: str, summary: str) -> None:
        self.done[node] = summary
        self.advance()

    def __rich__(self) -> Table:
        if self._cycle and time.perf_counter() - self._last_swap > self.SWAP_SECONDS:
            self._chatter = next(self._cycle)
            self._last_swap = time.perf_counter()

        t = Table.grid(padding=(0, 1))
        t.add_column(width=2)
        t.add_column(width=38)
        t.add_column(style=C_DIM)

        # Someone else owns the screen -- the interview. Draw nothing at all
        # rather than trying to coexist.
        if render.is_muted():
            return t

        if self.compact:
            if not self._current:
                return t
            st = BY_NODE[self._current]
            t.add_row(Spinner("dots", style=C_KEY),
                      f"[{C_KEY}]{st['label']}[/]",
                      f"[dim italic]{self._chatter}[/]  "
                      f"[dim]{self.elapsed:.0f}s[/]")
            return t

        for st in STAGES:
            node, label = st["node"], st["label"]
            if node in self.skipped:
                t.add_row("", f"[dim]{label}[/]", "[dim]skipped[/]")
            elif node in self.done:
                t.add_row(f"[{C_OK}]\u2713[/]", label, self.done[node])
            elif node == self._current:
                t.add_row(Spinner("dots", style=C_KEY),
                          f"[{C_KEY}]{label}[/]",
                          f"[dim italic]{self._chatter}[/]  "
                          f"[dim]{self.elapsed:.0f}s[/]")
            else:
                t.add_row("", f"[dim]{label}[/]", "")
        return t

    def _row(self, t: Table, node: str, label: str, running: bool = False) -> None:
        if node in self.skipped:
            t.add_row("", f"[dim]{label}[/]", "[dim]skipped[/]")
        elif node in self.done:
            t.add_row(f"[{C_OK}]\u2713[/]", label, self.done[node])
        elif running:
            t.add_row(f"[{C_KEY}]\u00b7[/]", f"[{C_KEY}]{label}[/]",
                      "[dim]in progress[/]")
        else:
            t.add_row("", f"[dim]{label}[/]", "")

    def commit_new(self, include_running: bool = False) -> Table | None:
        """The stages that have finished since the last commit, ready to print.

        Returns None when nothing new has finished, so a caller can skip
        printing an empty block. Marks what it returns as committed, so the same
        line is never printed twice.
        """
        t = Table.grid(padding=(0, 1))
        t.add_column(width=2)
        t.add_column(width=38)
        t.add_column(style=C_DIM)
        rows = 0
        for st in STAGES:
            node = st["node"]
            if node in self._committed:
                continue
            if node in self.done or node in self.skipped:
                self._row(t, node, st["label"])
                self._committed.add(node)
                rows += 1
            elif include_running and node == self._current:
                self._row(t, node, st["label"], running=True)
                rows += 1
        return t if rows else None

    def snapshot(self) -> Table | None:
        """What to freeze into the scrollback before someone writes under it."""
        return self.commit_new(include_running=True)

    def final(self) -> Table | None:
        """Whatever has not been printed yet, once the graph has returned."""
        self._current = None
        self._cycle = None
        self.compact = False
        return self.commit_new()


class ScenarioSession:
    def __init__(self, use_rag: bool = True, run_sdlc: bool = True,
                 classifier: str = "auto", interactive: bool = True,
                 run_analysis: bool = True):
        self.use_rag = use_rag
        self.run_sdlc = run_sdlc
        self.run_analysis = run_analysis
        self.classifier = classifier
        # Off for one-shot and scripted runs: a batch loop or a --json eval
        # must never stop for input it will never get.
        self.interactive = interactive
        self.detail = False
        # Temporary stores holding documents attached with /docs. Searched
        # alongside the curated corpus; empty for a normal session.
        self.extra_dirs: list = []
        self.last_state: dict | None = None

    def handle(self, text: str) -> dict:
        from main import needs_review
        from src.graph import build_pipeline

        skipped: set[str] = set()
        if not self.use_rag:
            skipped |= {"retrieval", "grounding"}
        if not self.run_sdlc:
            skipped.add("sdlc")
        if not self.run_analysis:
            skipped |= {"conflicts", "compliance", "security", "risk", "clarification"}

        order = [st["node"] for st in STAGES if st["node"] not in skipped]
        prog = _Progress(order, skipped)
        state: dict = {
            "text": text, "requirements": [], "quality_report": None,
            "classifier": self.classifier, "use_rag": self.use_rag,
            "context": [], "grounding": {}, "sdlc": None,
            "run_sdlc": self.run_sdlc,
            # The SDLC node asks its follow-up questions only when this is set,
            # and it asks them BEFORE scoring, so one recommendation is printed
            # rather than one that is immediately superseded.
            "interactive": self.interactive,
            "extra_dirs": self.extra_dirs,
            "run_analysis": self.run_analysis,
            "run_elaboration": True,
        }

        live = Live(prog, console=console, refresh_per_second=8, transient=True)
        live.start()
        # Tell the rest of the app who owns the terminal, so the interview --
        # which fires from deep inside the graph and cannot reach this object --
        # can quiet it before prompting.
        render.set_live(live, prog)
        try:
            for chunk in build_pipeline().stream(state, stream_mode="updates"):
                for node, update in chunk.items():
                    state.update(update or {})
                    st = BY_NODE.get(node)
                    if st:
                        try:
                            prog.finish(node, st["summary"](state))
                        except Exception:
                            prog.finish(node, "")

                    # Print what is finished before resuming the spinner, so a
                    # reader is reading results while the rest still runs.
                    milestone = MILESTONES.get(node)
                    if milestone:
                        live.stop()
                        # The lines that just finished, printed once and left
                        # alone -- then the result underneath them. The list
                        # grows downward instead of redrawing whole.
                        newly = prog.commit_new()
                        if newly is not None:
                            console.print(newly)
                        self._emit(milestone, state, prog.elapsed)
                        if any(n not in prog.done for n in order):
                            prog.compact = True
                            live = Live(prog, console=console,
                                        refresh_per_second=8, transient=True)
                            live.start()
                            render.set_live(live, prog)
        finally:
            try:
                live.stop()
            except Exception:
                pass
            render.set_live(None)

        tail = prog.final()
        if tail is not None:
            console.print(tail)
        console.print(f"[dim]done in {prog.elapsed:.0f}s[/]")

        render.print_review(needs_review(state))
        self.last_state = state
        return state

    def _emit(self, what: str, state: dict, elapsed: float) -> None:
        if what == "requirements":
            render.print_requirements(state.get("requirements", []),
                                      state.get("grounding", {}), elapsed)
        elif what == "quality":
            render.print_quality(state.get("quality_report"))
        elif what == "analysis":
            from src.cli.render_analysis import print_analysis

            print_analysis(state)
        elif what == "sdlc":
            render.print_sdlc(state.get("sdlc"), detail=self.detail)
