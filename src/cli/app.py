"""The interactive application: a REPL over the pipeline.

Line-based rather than full-screen, deliberately. A run emits a long report
worth scrolling back through, and a full-screen TUI would trap it in a pane.
This is the same shape as the agent CLIs it is modelled on.

Input is `prompt_toolkit`: multi-line by default (Enter inserts a newline,
**Esc-Enter or Alt-Enter submits**), because a problem scenario is a paragraph
and losing it to a stray Enter is infuriating. Slash commands complete on Tab
and submit on Enter alone, since those are always one line.

Everything expensive happens once at startup -- the classifier load and the LLM
pin are ~96% of a cold run's wall-clock (Day 13), so an interactive session pays
it a single time instead of per scenario.
"""

from __future__ import annotations

import os
from pathlib import Path

# Before prompt_toolkit or anything that pulls in transformers: the progress
# bars and hub warnings otherwise print into the middle of the application.
from src.cli.quiet import silence

silence()

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style

from src.cli import render
from src.cli.modes import DEFAULT_MODE, MODES, ModeUnavailable
from src.cli.render import C_BAD, C_DIM, C_KEY, C_OK, C_WARN, console

HISTORY = Path.home() / ".local/share/syntheia/history"

COMMANDS = {
    "/help":      "show this",
    "/mode":      "switch mode: /mode scenario | interview | chat",
    "/modes":     "list modes and what they do",
    "/docs":      "attach documents to cite from: /docs file.pdf | /docs | /docs clear",
    "/load":      "run a scenario from a file: /load path/to/scenario.txt",
    "/rag":       "grounding on or off: /rag on | off",
    "/sdlc":      "phase 2 on or off: /sdlc on | off",
    "/analysis":  "conflict/compliance/security/risk on or off: /analysis on | off",
    "/save":      "write the last run as JSON: /save run.json",
    "/export":    "write the SRS (Word + Markdown) for the last run: /export out/ [iso29148]",
    "/why":       "explain one framework answer: /why PTR-2",
    "/detail":    "show or hide the framework working: /detail on | off",
    "/clear":     "clear the screen",
    "/exit":      "quit (also Ctrl-D)",
}

PT_STYLE = Style.from_dict({
    "prompt": "ansicyan bold",
    "mode": "ansimagenta",
    "flag": "ansiyellow",
})


class App:
    def __init__(self, use_rag: bool = True, run_sdlc: bool = True,
                 classifier: str = "auto", run_analysis: bool = True):
        self.use_rag = use_rag
        self.run_sdlc = run_sdlc
        self.run_analysis = run_analysis
        self.classifier = classifier
        self.mode_name = DEFAULT_MODE
        self.detail = False
        # Attached for the whole session, not one run: you attach a regulation
        # once and try several scenarios against it, which is the actual
        # workflow. Rebuilt whenever the list changes, dropped on exit.
        self.doc_paths: list[Path] = []
        self.corpus = None
        self.session_obj = None
        self.last_state: dict | None = None
        HISTORY.parent.mkdir(parents=True, exist_ok=True)

        kb = KeyBindings()

        @kb.add("enter")
        def _(event):
            """Enter submits. Always.

            The first version tried to be clever -- submit a single line, insert
            a newline once the buffer was already multi-line -- and it trapped
            you: after one Enter the buffer contained a newline, so every
            subsequent Enter added another and there was no way to submit at
            all except a shortcut nobody had been told about.

            Pasting a paragraph still works, because a bracketed paste inserts
            its newlines directly and never fires this binding. Typing one by
            hand uses Alt-Enter. Enter meaning "send" is what every other prompt
            in the world does, and guessing on the user's behalf was the bug.
            """
            buf = event.current_buffer
            if buf.text.strip():
                buf.validate_and_handle()
            # Empty or whitespace-only: swallow it rather than submitting a
            # blank line and reprinting the banner.

        @kb.add("escape", "enter")   # Alt-Enter in most terminals
        @kb.add("c-j")               # Ctrl-J, for terminals that eat Alt-Enter
        def _(event):
            event.current_buffer.insert_text("\n")

        self.prompt = PromptSession(
            history=FileHistory(str(HISTORY)),
            completer=WordCompleter(list(COMMANDS) + list(MODES),
                                    sentence=True),
            key_bindings=kb,
            multiline=True,
            style=PT_STYLE,
            complete_while_typing=False,
        )

    # ── plumbing ───────────────────────────────────────────────────────

    def _provider(self) -> str:
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        where = "cloud" if "localhost" not in host and "127.0.0.1" not in host else "local"
        return f"{model} [{where}]"

    def _classifier_label(self) -> str:
        from src.agents.classification import resolve_model_dir

        if self.classifier not in ("auto", "bert"):
            return f"{self.classifier} LLM baseline"
        d = resolve_model_dir()
        return f"fine-tuned ({d.name})" if d else "few-shot LLM (none trained)"

    def _ensure_session(self):
        mode = MODES[self.mode_name]
        if self.session_obj is None:
            self.session_obj = mode.start(
                use_rag=self.use_rag, run_sdlc=self.run_sdlc,
                classifier=self.classifier)
        else:
            self.session_obj.use_rag = self.use_rag
            self.session_obj.run_sdlc = self.run_sdlc
        # Set, not passed: other modes share `start(**kw)` and do not take it.
        if hasattr(self.session_obj, "run_analysis"):
            self.session_obj.run_analysis = self.run_analysis
        self.session_obj.detail = self.detail
        self._sync_session()
        return self.session_obj

    def _rprompt(self) -> HTML:
        bits = []
        if not self.use_rag:
            bits.append("no-rag")
        if not self.run_sdlc:
            bits.append("no-sdlc")
        if not self.run_analysis:
            bits.append("no-analysis")
        return HTML(f"<flag>{'  '.join(bits)}</flag>") if bits else HTML("")

    # ── commands ───────────────────────────────────────────────────────

    def cmd(self, line: str) -> bool:
        """Handle a slash command. Returns False to quit."""
        parts = line.strip().split()
        name, arg = parts[0], (parts[1] if len(parts) > 1 else "")

        if name in ("/exit", "/quit", "/q"):
            return False

        if name == "/help":
            t = render.Table(box=None, padding=(0, 2), show_header=False)
            t.add_column(style=C_KEY)
            t.add_column(style=C_DIM)
            for c, d in COMMANDS.items():
                t.add_row(c, d)
            console.print(render.Panel(
                t, title="[bold]commands[/]", border_style=C_DIM,
                subtitle="[dim]Enter sends · Alt-Enter (or Ctrl-J) for a newline · paste works as-is[/]",
                padding=(1, 2)))

        elif name == "/modes":
            t = render.Table(box=None, padding=(0, 2), show_header=False)
            t.add_column(style=C_KEY)
            t.add_column(width=10)
            t.add_column(style=C_DIM)
            for k, m in MODES.items():
                mark = f"[{C_OK}]●[/]" if k == self.mode_name else " "
                state = "ready" if m.available else f"[{C_DIM}]not built[/]"
                t.add_row(f"{mark} {k}", state, m.summary)
            console.print(render.Panel(t, title="[bold]modes[/]",
                                       border_style=C_DIM, padding=(1, 2)))

        elif name == "/mode":
            if arg not in MODES:
                console.print(f"[{C_BAD}]unknown mode {arg!r}[/] "
                              f"[dim]— try /modes[/]")
            elif not MODES[arg].available:
                console.print(render.Panel(
                    MODES[arg].placeholder,
                    title=f"[{C_WARN}]{arg} — not built yet[/]",
                    border_style=C_DIM, padding=(0, 1)))
            else:
                self.mode_name, self.session_obj = arg, None
                console.print(f"[{C_OK}]mode: {arg}[/]")

        elif name == "/rag":
            self.use_rag = arg != "off"
            console.print(f"[dim]grounding "
                          f"{'on' if self.use_rag else 'OFF (control arm)'}[/]")

        elif name == "/sdlc":
            self.run_sdlc = arg != "off"
            console.print(f"[dim]phase 2 {'on' if self.run_sdlc else 'off'}[/]")

        elif name == "/analysis":
            self.run_analysis = arg != "off"
            console.print(f"[dim]conflict/compliance/security/risk analysis "
                          f"{'on' if self.run_analysis else 'off'}[/]")

        elif name == "/detail":
            self.detail = arg != "off"
            console.print(f"[dim]framework working "
                          f"{'shown' if self.detail else 'hidden'}[/]")

        elif name == "/docs":
            self._docs(arg)

        elif name == "/clear":
            console.clear()

        elif name == "/load":
            if not arg:
                console.print(f"[{C_BAD}]usage: /load path/to/scenario.txt[/]")
            else:
                p = Path(arg)
                if not p.is_file():
                    console.print(f"[{C_BAD}]no such file: {p}[/]")
                else:
                    console.print(f"[dim]loaded {p}[/]")
                    self._run(p.read_text(encoding="utf-8").strip())

        elif name == "/save":
            if not self.last_state:
                console.print(f"[{C_BAD}]nothing to save yet[/]")
            elif not arg:
                console.print(f"[{C_BAD}]usage: /save run.json[/]")
            else:
                from main import write_json

                write_json(Path(arg), None, self.last_state.get("text", ""),
                           self.last_state)
                console.print(f"[{C_OK}]wrote {arg}[/]")

        elif name == "/export":
            if not self.last_state or not self.last_state.get("requirements"):
                console.print(f"[{C_BAD}]nothing to export yet — run a scenario first[/]")
            elif not arg:
                console.print(f"[{C_BAD}]usage: /export out_dir [ieee830 | iso29148 | both][/]")
            else:
                from main import run_payload
                from src.generators.srs import generate_srs

                layout = parts[2] if len(parts) > 2 else "ieee830"
                if layout not in ("ieee830", "iso29148", "both"):
                    console.print(f"[{C_BAD}]layout must be ieee830, iso29148 or both[/]")
                else:
                    payload = run_payload(None, self.last_state.get("text", ""),
                                          self.last_state)
                    paths = generate_srs(payload, Path(arg), layout=layout, verbose=False)
                    for path in paths.values():
                        console.print(f"[{C_OK}]wrote {path}[/]")

        elif name == "/why":
            self._why(arg)

        else:
            console.print(f"[{C_BAD}]unknown command {name!r}[/] [dim]— /help[/]")
        return True

    def _docs(self, arg: str) -> None:
        """Attach, list or clear the documents this session can cite from."""
        if arg == "clear":
            self._drop_corpus()
            self.doc_paths = []
            console.print(f"[dim]detached — citing from the curated corpus only[/]")
            return

        if not arg:
            if not self.doc_paths:
                console.print("[dim]no documents attached. "
                              "/docs path/to/file.pdf to add one[/]")
                return
            t = render.Table(box=None, padding=(0, 2), show_header=False)
            t.add_column(style=C_KEY)
            t.add_column(style=C_DIM)
            for name, msg in (self.corpus.report if self.corpus else []):
                t.add_row(name, msg)
            console.print(render.Panel(
                t, title="[bold]attached for this session[/]",
                subtitle=r"[dim]citations from these are marked \[supplied][/]",
                border_style=C_DIM, padding=(1, 2)))
            return

        path = Path(arg).expanduser()
        if not path.exists():
            console.print(f"[{C_BAD}]no such file or directory: {path}[/]")
            return

        already = {n for n, _ in (self.corpus.report if self.corpus else [])}
        self.doc_paths.append(path)
        self._rebuild_corpus(quiet_for=already)

    def _rebuild_corpus(self, quiet_for: set[str] | None = None) -> None:
        """Re-index everything attached. Cheap enough to redo on each addition.

        Rebuilding rather than appending because the store is disposable and
        incremental state is where this would rot -- a half-added document is
        harder to reason about than a rebuild that takes a second.
        """
        from src.rag.session import SessionCorpus, SuppliedDocError

        self._drop_corpus()
        try:
            with render.Thinking(["reading the document",
                                  "splitting it into clauses",
                                  "embedding the passages",
                                  "indexing for this session"]):
                self.corpus = SessionCorpus(self.doc_paths).build()
        except SuppliedDocError as e:
            console.print(f"[{C_BAD}]{e}[/]")
            self.doc_paths.pop()
            return

        # Report only what is new. Everything is re-indexed on each addition,
        # but re-announcing documents already attached reads like they were
        # added twice.
        seen = quiet_for or set()
        for name, msg in self.corpus.report:
            if name in seen:
                continue
            ok = "passages" in msg
            console.print(f"  [{C_OK if ok else C_BAD}]{'+' if ok else '!'}[/] "
                          f"{name} [dim]— {msg}[/]")
        self._sync_session()

    def _drop_corpus(self) -> None:
        if self.corpus:
            self.corpus.cleanup()
        self.corpus = None
        self._sync_session()

    def _sync_session(self) -> None:
        dirs = ([self.corpus.persist_dir]
                if self.corpus and self.corpus.usable else [])
        if self.session_obj is not None:
            self.session_obj.extra_dirs = dirs

    def _why(self, qid: str) -> None:
        rec = (self.last_state or {}).get("sdlc")
        if not rec:
            console.print(f"[{C_BAD}]no analysis in memory — run a scenario first[/]")
            return
        if not qid:
            console.print(f"[{C_BAD}]usage: /why PTR-2[/]")
            return
        a = next((x for x in rec.criteria
                  if x.question_id.lower() == qid.lower()), None)
        if not a:
            console.print(f"[{C_BAD}]no answer for {qid!r} in the last run[/]")
            return

        from src.sdlc.framework import Framework

        q = Framework.load().question(a.question_id)
        body = render.Text()
        body.append(f"{q.text}\n", style="bold")
        if q.gloss:
            body.append(f"{q.gloss}\n\n", style=C_DIM)
        body.append(f"answered: {a.answer}\n")
        body.append(f"because:  {a.reasoning}\n", style=C_DIM)
        body.append(f"evidence: {', '.join(a.evidence) or 'none'}\n", style=C_DIM)
        if q.is_derived:
            body.append("\nthis question is ours, not the supplied framework's",
                        style="yellow")
        console.print(render.Panel(body, title=f"[bold]{a.question_id}[/]",
                                   border_style=C_KEY, padding=(1, 2)))

    # ── loop ───────────────────────────────────────────────────────────

    def _run(self, text: str) -> None:
        render.echo_input(text, MODES[self.mode_name].prompt)
        try:
            self.last_state = self._ensure_session().handle(text)
        except ModeUnavailable as e:
            console.print(f"[{C_BAD}]{e}[/]")
        except KeyboardInterrupt:
            console.print(f"\n[{C_DIM}]interrupted[/]")

    def run(self) -> int:
        from src.utils.warmup import warm

        console.print()
        render.banner(self._provider(), self._classifier_label(),
                      "on" if self.use_rag else "OFF (--no-rag)", self.mode_name)
        warm()

        while True:
            try:
                line = self.prompt.prompt(
                    HTML(f"\n<prompt>{MODES[self.mode_name].prompt} ❯</prompt> "),
                    rprompt=self._rprompt(),
                )
            except KeyboardInterrupt:
                continue                       # Ctrl-C clears the line
            except EOFError:
                break                          # Ctrl-D quits

            text = line.strip()
            if not text:
                continue
            if text.startswith("/"):
                if not self.cmd(text):
                    break
                continue
            self._run(text)

        self._drop_corpus()          # temporary stores do not outlive the session
        console.print("[dim]bye[/]")
        return 0
