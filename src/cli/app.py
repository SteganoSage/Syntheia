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
    "/load":      "run a scenario from a file: /load path/to/scenario.txt",
    "/rag":       "grounding on or off: /rag on | off",
    "/sdlc":      "phase 2 on or off: /sdlc on | off",
    "/save":      "write the last run as JSON: /save run.json",
    "/why":       "explain one framework answer: /why PTR-2",
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
                 classifier: str = "auto"):
        self.use_rag = use_rag
        self.run_sdlc = run_sdlc
        self.classifier = classifier
        self.mode_name = DEFAULT_MODE
        self.session_obj = None
        self.last_state: dict | None = None
        HISTORY.parent.mkdir(parents=True, exist_ok=True)

        kb = KeyBindings()

        @kb.add("enter")
        def _(event):
            """Submit if it is a command or a single line; else newline.

            A scenario is a paragraph, so Enter has to be able to mean
            "newline". But requiring Esc-Enter for `/help` would be obnoxious,
            so anything starting with `/` submits immediately, as does a single
            line that looks finished.
            """
            buf = event.current_buffer
            text = buf.text
            if text.lstrip().startswith("/") or not text.strip():
                buf.validate_and_handle()
            elif "\n" in text:
                buf.insert_text("\n")
            else:
                buf.validate_and_handle()

        @kb.add("escape", "enter")
        @kb.add("c-j")
        def _(event):
            event.current_buffer.validate_and_handle()

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
        return self.session_obj

    def _rprompt(self) -> HTML:
        bits = []
        if not self.use_rag:
            bits.append("no-rag")
        if not self.run_sdlc:
            bits.append("no-sdlc")
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
                subtitle="[dim]Enter submits · Esc-Enter for a newline mid-paragraph[/]",
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

        elif name == "/why":
            self._why(arg)

        else:
            console.print(f"[{C_BAD}]unknown command {name!r}[/] [dim]— /help[/]")
        return True

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

        console.print("[dim]bye[/]")
        return 0
