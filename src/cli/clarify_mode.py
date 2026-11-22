"""Put the clarification questions to a person at the terminal.

Same conventions as `interview_mode`: the exchange takes the terminal (the
pipeline checklist is suspended so it cannot repaint over the prompt), the
questions go out as one numbered list rather than a form, and the answer is
free text -- a person answers the way they talk, and the clarification agent's
job is to read that, not to make them fill fields.

Never raises. Ctrl-C or Ctrl-D skips, and the questions stay open in the
report: an unanswered question is still worth listing.
"""

from __future__ import annotations

from rich.panel import Panel
from rich.text import Text

from src.cli.render import C_DIM, C_KEY, console, suspended


def ask_clarifications(questions) -> str:
    if not questions:
        return ""
    with suspended(commit=True):
        body = Text()
        body.append("Some things in this project are unclear or missing. Answer any "
                    "you can, in your\nown words — refer to the numbers if it helps. "
                    "Anything you skip stays on the open list.\n\n", style=C_DIM)
        for n, q in enumerate(questions, start=1):
            body.append(f"{n}. ", style=C_KEY)
            body.append(f"{q.question}\n")
            body.append(f"   {q.why}\n", style=C_DIM)
        console.print(Panel(body, title="[bold]before I finish[/]",
                            border_style=C_KEY, padding=(1, 2)))

        from prompt_toolkit import PromptSession
        from prompt_toolkit.formatted_text import HTML

        console.print("  [dim](Enter to skip)[/]")
        try:
            return PromptSession().prompt(HTML("  <b>❯</b> ")).strip()
        except (KeyboardInterrupt, EOFError):
            console.print(f"\n  [{C_DIM}]skipped[/]")
            return ""
