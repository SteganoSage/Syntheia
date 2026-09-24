"""Ask about what the scenario left out — in plain language, before deciding.

Three design rules, each fixing a fault in the first version.

**Ask in prose, not in the rubric's words.** The framework's questions are a
scoring instrument, not an interview script. "Are the majority of team members
new to the technology domain?" is a strange thing to be asked and invites a
yes/no that flattens whatever the person actually knows. So the questions are
grouped by table, turned into one request for information, and answered however
the person likes. Their reply is then run through the SAME inference and the
SAME evidence gate the scenario goes through -- an answer they did not give
stays unanswered, exactly as before.

**Ask before deciding, not after.** The first version printed a recommendation,
then asked, then printed a different one. That shows the user a result it is
about to retract, and invites them to treat the first answer as real. Gaps are
now found and filled while the recommendation is still being computed, and one
result is produced.

**Never say what an answer would change.** Showing "yes leads to Prototype, no
leads to Agile" is not transparency, it is leading the witness: it lets the
answer be chosen for the outcome rather than for the truth, which quietly turns
ground truth into a preference. The stakes are computed to decide WHICH
questions are worth asking; they are not shown to the person answering them.
"""

from __future__ import annotations

from rich.text import Text

from src.cli.render import (C_DIM, C_KEY, C_OK, Panel, Thinking, console,
                            suspended)

MAX_TABLES = 2          # two prompts is a conversation; four is a form

# Shown while the reply is being read. Playful, but scoped to the step -- the
# spinner is the only thing on screen while a blocking call runs, and silence
# there is indistinguishable from a crash.
READING = [
    "decoding your typing",
    "reading between the lines",
    "'maybe' absolutely counts, relax",
    "translating human into framework",
    "you said a lot there",
    "cross-referencing with what you actually wrote",
    "no, it will not put words in your mouth",
    "squinting at the word 'mostly'",
    "taking you at your word",
]

# What to actually say, per table. Written as a request for information rather
# than a quiz, because that is what gets a usable answer.
TABLE_PROMPTS = {
    "requirements": (
        "the requirements themselves",
        "How well understood are they? Are they likely to change as work goes "
        "on, and does anyone need to see something working before they can pin "
        "them down?"),
    "project_type_risk": (
        "the project and its risks",
        "Is this new ground for the organisation or an extension of something "
        "that exists? Does it have to work with other systems? Is the budget "
        "settled, and how long will this product be around?"),
    "project_team": (
        "the team",
        "How many people, and how familiar are they with this kind of problem "
        "and this technology? Will they stay on it throughout, and is training "
        "available if they need it?"),
    "user_community": (
        "the users",
        "Who will represent them, how much of their time will the project get, "
        "how well do they know the business area, and how involved do they "
        "want to be?"),
}


def _prompt_for(table_id: str, questions) -> tuple[str, str]:
    subject, ask = TABLE_PROMPTS.get(
        table_id, ("the project", "Tell me what you can."))
    return subject, ask


def gather(framework, pivotal: list[dict]) -> tuple[str, list[str]]:
    """Put the gaps to the user in prose. Returns what they wrote and what was asked.

    Never raises: Ctrl-C or Ctrl-D abandons the interview and returns what has
    been said so far, because a partly-informed decision beats a fabricated one.
    """
    if not pivotal:
        return "", []

    by_table: dict[str, list] = {}
    for item in pivotal:
        by_table.setdefault(item["question"].table, []).append(item["question"])

    # Most-uncertain tables first, so a person who answers one prompt and stops
    # has answered the one that mattered most.
    tables = sorted(by_table.items(), key=lambda kv: -len(kv[1]))[:MAX_TABLES]

    intro = Text()
    intro.append("The scenario does not say enough to settle this.\n", style=C_DIM)
    intro.append("Tell me what you know in your own words — a sentence or two "
                 "is plenty, and\nanything you leave out simply stays "
                 "unanswered.", style=C_DIM)
    console.print(Panel(intro, title="[bold]before I decide[/]",
                        border_style=C_KEY, padding=(1, 2)))

    from prompt_toolkit import PromptSession
    from prompt_toolkit.formatted_text import HTML

    session = PromptSession()
    replies: list[str] = []

    for table_id, questions in tables:
        subject, ask = _prompt_for(table_id, questions)
        console.print()
        console.print(f"  [bold]About {subject}[/]")
        console.print(f"  [dim]{ask}[/]")
        console.print(f"  [dim](Enter to skip)[/]")
        try:
            said = session.prompt(HTML("  <b>❯</b> ")).strip()
        except (KeyboardInterrupt, EOFError):
            console.print(f"\n  [{C_DIM}]skipping the rest[/]")
            break
        if said:
            replies.append(said)

    return "\n".join(replies), [tid for tid, _ in tables]


def answers_from(reply: str, framework, questions) -> dict[str, bool]:
    """Read framework answers out of what the person wrote.

    Uses `infer_from_reply`, which applies the same evidence gate: an
    answer it cannot trace back to something the person actually said comes back
    not_stated and is left unscored. That is the whole reason to reuse it rather
    than parse here -- the standard applied to the user's words is the same one
    applied to the scenario, so a confident-sounding interview cannot buy a
    looser rule than a confident-sounding brief.
    """
    if not reply.strip() or not questions:
        return {}

    from src.agents.sdlc import infer_from_reply

    # Reading a reply is one LLM call per table -- seconds on the 8B, longer on
    # the 3B -- and it lands immediately after the person has typed, which is
    # exactly the moment an app looks broken if it says nothing.
    with Thinking(READING):
        answered = infer_from_reply(reply, questions)
    return {a.question_id: a.as_bool for a in answered if a.as_bool is not None}


def report(framework, supplied: dict[str, bool], asked: int) -> None:
    """Say what was understood, so a misreading is visible rather than silent."""
    if not supplied:
        if asked:
            console.print(f"  [{C_DIM}]nothing further could be established — "
                          f"deciding on the scenario alone[/]")
        return
    body = Text()
    for qid, val in supplied.items():
        q = framework.question(qid)
        body.append("• ", style=C_OK)
        body.append(f"{q.text if q else qid}\n")
        body.append(f"    understood as: {'yes' if val else 'no'}\n", style=C_DIM)
    console.print(Panel(body, title="[bold]what I took from that[/]",
                        border_style=C_OK, padding=(1, 2)))


def fill_gaps(framework, answer_map: dict[str, bool],
              weak: set[str] | None = None) -> dict[str, bool]:
    """The whole interview, as one call. Returns answers to ADD to the map.

    Invoked from inside `recommend_sdlc`, before scoring, so the person sees one
    recommendation rather than one that is superseded a moment later.
    """
    pivotal = framework.pivotal(answer_map, weak=weak)

    # Only ask about what a person can actually add. The Requirements table and
    # most of Project Type & Risk are read OFF the requirements, which are
    # printed a few lines above -- asking "how well understood are they?" is
    # admitting the inference failed, not filling a gap, and it invites the user
    # to grade work they just watched happen.
    #
    # What genuinely needs a human: the supplied tables (team, users), and the
    # two questions marked evidence_from: scenario, because no requirement can
    # testify about a budget or how long a product will live.
    pivotal = [d for d in pivotal
               if d["question"].source == "supplied"
               or d["question"].evidence_from == "scenario"]
    if not pivotal:
        return {}

    # The whole exchange takes the terminal. Without this the pipeline
    # checklist keeps repainting on its own thread, straight over the question
    # and over what the person is typing.
    # commit=True: freeze the finished stages into the scrollback first, so the
    # questions are written under a checklist that stays put rather than one
    # that disappears and then redraws itself whole below the conversation.
    with suspended(commit=True):
        reply, tables = gather(framework, pivotal)

    # Read EVERY open question in the tables that were asked about, not just the
    # pivotal ones. The prompt is broad on purpose -- "how many people, how
    # familiar are they, will they stay, is training available" -- so a reply
    # answers four things at once, and passing only the pivotal question threw
    # three of them away. One run answered all five team questions and this
    # recorded one.
    #
    # The extras are not wasted even when they do not move this decision: they
    # raise coverage, and a combination of them can settle what no single
    # question could.
        weak = weak or set()
        open_qs = [q for q in framework.questions
                   if q.table in tables
                   and (q.id not in answer_map or q.id in weak)]

        supplied = answers_from(reply, framework, open_qs)
        report(framework, supplied, asked=len(pivotal))
    return supplied
