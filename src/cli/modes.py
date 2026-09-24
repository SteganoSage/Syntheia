"""What the application can be asked to do, and the seam new abilities plug into.

Right now there is one real mode: paste a problem scenario, get grounded
requirements and an SDLC recommendation. Two more are coming and they are
genuinely different shapes of interaction, not variations on a prompt:

    scenario    one-shot.   text in -> full analysis out
    interview   multi-turn. the system asks the questions, because a 50-word
                brief cannot answer 20 framework questions and the honest fix
                is to ask rather than to infer harder
    chat        free-form.  questions about a completed analysis, or about
                requirements engineering generally

The seam exists now so those land as a new file rather than as `if mode ==`
scattered through the REPL. The unbuilt ones are registered as `available =
False` and say so when selected -- a mode that half works is worse than a mode
that admits it does not exist yet.

Why `interview` matters more than it sounds: the Project Team and User Community
tables are answerable only from facts a problem statement rarely carries. Across
five scenarios the model fabricated them rather than leave them blank, and the
evidence gate now drops those answers -- which is correct, and leaves 9 of 20
questions permanently unscored. Asking the user four questions closes that
properly. It is the designed home for a problem we currently only suppress.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol


class ModeSession(Protocol):
    """One mode's handling of input. Stateful: `interview` and `chat` need it."""

    def handle(self, text: str) -> None: ...


@dataclass(frozen=True)
class Mode:
    name: str
    summary: str                      # one line, shown by /help and /mode
    prompt: str                       # what the input line says
    factory: Callable[..., ModeSession] | None = None
    available: bool = True
    placeholder: str = ""             # shown when an unbuilt mode is selected

    def start(self, **kw) -> ModeSession:
        if not self.available or self.factory is None:
            raise ModeUnavailable(self.name, self.placeholder)
        return self.factory(**kw)


class ModeUnavailable(RuntimeError):
    def __init__(self, name: str, detail: str = ""):
        self.name, self.detail = name, detail
        super().__init__(detail or f"mode '{name}' is not built yet")


def _scenario_factory(**kw):
    from src.cli.scenario_mode import ScenarioSession

    return ScenarioSession(**kw)


MODES: dict[str, Mode] = {
    "scenario": Mode(
        name="scenario",
        summary="paste a problem scenario -> requirements, citations, SDLC",
        prompt="scenario",
        factory=_scenario_factory,
    ),
    # Built, but not a mode you switch to -- it fires by itself from inside
    # scenario mode when the framework says an open question would change the
    # recommendation, and stays quiet when nothing would. Listed here so /modes
    # explains where it went rather than leaving a gap someone goes looking for.
    "interview": Mode(
        name="interview",
        summary="automatic — asks only when an open question would change the answer",
        prompt="interview",
        available=False,
        placeholder=(
            "This is not a mode any more. It happens on its own: after the "
            "requirements and quality report are in, the framework is asked "
            "which unanswered questions could still change the recommendation. "
            "If any could, you are asked about them in plain English before a "
            "result is shown; if none could, nothing interrupts you."
        ),
    ),
    "chat": Mode(
        name="chat",
        summary="ask questions about the last analysis, or about RE generally",
        prompt="chat",
        available=False,
        placeholder=(
            "Not built yet. Intended for interrogating a completed analysis "
            "-- 'why did PTR-2 come back yes?', 'what would change if the team "
            "were larger?' -- against the run already in memory."
        ),
    ),
}

DEFAULT_MODE = "scenario"
