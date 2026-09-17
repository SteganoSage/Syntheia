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
# Every line has to be something the stage genuinely does -- filler that turns
# out to be wrong is worse than no filler.
STAGES: list[dict] = [
    dict(node="retrieval", label="retrieving source material",
         summary=lambda s: _passages(s),
         chatter=["searching the knowledge base",
                  "ranking clauses by relevance",
                  "checking the scenario clears the relevance floor",
                  "reading RBI master directions",
                  "skipping tables of contents"]),
    dict(node="extraction", label="extracting requirements",
         summary=lambda s: f"{len(s.get('requirements') or [])} requirements",
         chatter=["reading the scenario",
                  "separating system requirements from project facts",
                  "writing acceptance criteria",
                  "naming stakeholders",
                  "deciding what the system must enforce",
                  "resisting the urge to invent a portal"]),
    dict(node="grounding", label="grounding against the knowledge base",
         summary=lambda s: _grounding(s),
         chatter=["matching requirements to clauses",
                  "resolving clause ids",
                  "checking named instruments actually exist",
                  "separating regulations from tender precedent",
                  "flagging claims the corpus cannot corroborate"]),
    dict(node="classification", label="classifying",
         summary=lambda s: _categories(s),
         chatter=["running the fine-tuned RoBERTa",
                  "applying the calibrated threshold",
                  "assigning multi-label categories"]),
    dict(node="quality", label="checking quality",
         summary=lambda s: _quality(s),
         chatter=["looking for unmeasurable adjectives",
                  "checking for undefined terminology",
                  "testing each requirement for ambiguity",
                  "ignoring its own citations as jargon"]),
    dict(node="sdlc", label="scoring the SDLC framework",
         summary=lambda s: (s["sdlc"].recommendation if s.get("sdlc") else "skipped"),
         chatter=["answering the Requirements table",
                  "answering Project Type & Risk",
                  "asking whether this is an integration project",
                  "checking the evidence is traceable",
                  "counting matches against the supervisor's tables",
                  "weighing Waterfall against V-Shaped",
                  "ranking without looking at the scores",
                  "writing the justification"]),
]

# Where to interrupt the spinner and print. Requirements are only complete once
# classification has filled `category`, so that is the earliest honest point --
# printing them after extraction would show every one with an empty category and
# then need reprinting.
BY_NODE = {st["node"]: st for st in STAGES}

MILESTONES = {
    "classification": "requirements",
    "quality": "quality",
    "sdlc": "sdlc",
}


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

    def final(self) -> Table:
        """The same checklist with nothing in flight, for scrollback."""
        self._current = None
        self._cycle = None
        return self.__rich__()


class ScenarioSession:
    def __init__(self, use_rag: bool = True, run_sdlc: bool = True,
                 classifier: str = "auto"):
        self.use_rag = use_rag
        self.run_sdlc = run_sdlc
        self.classifier = classifier
        self.last_state: dict | None = None

    def handle(self, text: str) -> dict:
        from main import needs_review
        from src.graph import build_pipeline

        skipped: set[str] = set()
        if not self.use_rag:
            skipped |= {"retrieval", "grounding"}
        if not self.run_sdlc:
            skipped.add("sdlc")

        order = [st["node"] for st in STAGES if st["node"] not in skipped]
        prog = _Progress(order, skipped)
        state: dict = {
            "text": text, "requirements": [], "quality_report": None,
            "classifier": self.classifier, "use_rag": self.use_rag,
            "context": [], "grounding": {}, "sdlc": None,
            "run_sdlc": self.run_sdlc,
        }

        live = Live(prog, console=console, refresh_per_second=8, transient=True)
        live.start()
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
                        self._emit(milestone, state, prog.elapsed)
                        if any(n not in prog.done for n in order):
                            live = Live(prog, console=console,
                                        refresh_per_second=8, transient=True)
                            live.start()
        finally:
            try:
                live.stop()
            except Exception:
                pass

        console.print(prog.final())
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
        elif what == "sdlc":
            render.print_sdlc(state.get("sdlc"))
