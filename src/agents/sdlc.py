"""SDLC recommendation agent — deliverable 2.

The supervisor's part 2: "feed those generated requirements back into an LLM and
ask it to identify the most appropriate SDLC, along with a proper software
engineering justification."

Three steps, and the middle one is not a model:

    1. INFER      LLM answers the framework's questions from the requirements,
                  citing requirement ids. Language judgement -- its strength.
    2. SCORE      Python counts matches against the supervisor's tables and
                  ranks the models. Arithmetic -- deterministic, inspectable,
                  and the reason the justification can show its working.
    3. JUSTIFY    LLM explains a decision that has already been made, citing
                  framework question ids and requirement ids.

Plus a cross-check: the model also ranks the options WITHOUT seeing the scores.
Agreement is evidence; disagreement is reported rather than resolved silently.

Everything runs at temperature 0. A recommendation engine that answers
differently on the same requirements cannot be defended, and the quality agent
already showed what sampling does to an analysis task.
"""

from __future__ import annotations

import re

from src.models.requirement import Requirement
from src.models.sdlc import (
    CriterionAnswer,
    CriterionAnswerSet,
    CrossCheckRanking,
    JustificationSet,
    SDLCRecommendation,
    ScoredModel,
    TableCoverage,
)
from src.sdlc.framework import Framework, Question
from src.utils.llm import call_structured

TOP_N = 3


def _format_statements(requirements: list[Requirement]) -> str:
    """Statements only -- no priority, risk or category.

    The Requirements table asks how well understood and how stable the
    requirements are. Priority and risk say nothing about either, but the model
    argues from them anyway: "high risk level implies frequent changes", "high
    priority implies a proof of concept is required". Three separate
    instructions not to do this were ignored, which is the same negative-
    constraint failure the quality agent showed.

    So the metadata is withheld for this table rather than forbidden. A rule you
    can enforce should not be delegated to a 3B model's compliance.
    """
    return "\n".join(f"[{r.id}] {r.statement}" for r in requirements)


def _format_requirements(requirements: list[Requirement]) -> str:
    """Every field the framework questions could need.

    The quality agent's whole failure was being sent a truncated requirement and
    then reporting that it was incomplete. Questions here ask about stability,
    integration and product longevity -- answerable only from risk level,
    regulations and dependencies, not from the statement alone.
    """
    # Field labels are marked as metadata, not left bare. With bare labels the
    # model read them as requirement content: one run answered REQ-1 with "the
    # requirements contain vague terms like 'priority', 'risk' and 'categories'"
    # -- which are this function's own column headings, not words in any
    # requirement.
    blocks = []
    for r in requirements:
        meta = [f"priority: {r.priority}", f"risk level: {r.risk_level}"]
        if r.category:
            meta.append(f"classified as: {', '.join(r.category)}")
        if r.applicable_regulations:
            meta.append(f"governed by: {', '.join(r.applicable_regulations)}")
        if r.dependencies:
            meta.append(f"depends on: {', '.join(r.dependencies)}")
        if r.assumptions:
            meta.append(f"assumes: {r.assumptions}")
        blocks.append(
            f"[{r.id}] {r.statement}\n"
            + "\n".join(f"      (metadata) {m}" for m in meta)
        )
    return "\n\n".join(blocks)


def _format_questions(questions: list[Question]) -> str:
    """Question text plus its gloss.

    The supervisor's wording is compressed to fit a slide, and the compression
    is where the inference errors came from. "Is the project a system
    integration project?" was answered NO for a platform doing real-time NSDL
    lookups, Aadhaar verification and video KYC -- read, plausibly, as "is a
    systems integrator doing the work". "Are the user representatives new to
    system definition?" was answered from a sentence about the development team.

    The wording is not changed -- it is the supervisor's rubric and the scoring
    depends on it. The gloss is an annotation beside it.
    """
    out = []
    for q in questions:
        out.append(f"{q.id}: {q.text}")
        if q.gloss:
            out.append(f"      ({q.gloss})")
    return "\n".join(out)


# ── 1. Infer the framework answers ─────────────────────────────────────

INFER_PROMPT = """\
You are a software engineering analyst. Given software requirements for a \
financial system, answer questions about the PROJECT those requirements imply.

Each question is answered true (yes) or false (no), about the project as a whole \
— not about any single requirement.

Rules:
- Reason from what the requirements IMPLY about the project, not only from what \
they literally state. A project that replicates an existing system has \
well-known requirements even though no requirement says "the requirements are \
well known".
- **Silence is not uncertainty.** If the requirements do not mention how often \
they will change, work it out from the project: a migration that adds no new \
functionality has stable requirements; a startup MVP does not. Do not answer \
"unknown, therefore unstable".
- Use what you know about financial systems. Core banking platforms, payment \
rails and KYC systems are long-lived infrastructure. A fixed delivery window \
with approved scope implies stable funding.
- **The original problem statement is given below the requirements. It is the \
best evidence for how the project will run** — it says things the requirements \
cannot, such as whether new functionality is being added or whether scope is \
expected to change. Read it first.
- Priority and risk level describe a requirement's importance, NOT how stable, \
clear or complete it is. A high-risk requirement can be perfectly well defined. \
Never argue from priority or risk to requirement stability.
- Cite the ONE or TWO requirements that actually decided the answer in \
`evidence`. Citing every requirement is not evidence and will be rejected.
- `reasoning` is one sentence naming the specific thing that decided it.
- Answer EVERY question given, using the exact question ids.
"""


def infer_criteria(requirements: list[Requirement],
                   questions: list[Question],
                   scenario: str = "") -> list[CriterionAnswer]:
    """Answer the framework questions, one table at a time.

    Batched per table rather than all ten at once. Asking for ten answers in a
    single output produced the same reasoning recycled across unrelated
    questions and evidence lists naming every requirement -- the model spreading
    itself thin across one long generation. A table is five related questions
    about one aspect of the project, which is a question a 3B can hold in mind.
    """
    by_table: dict[str, list[Question]] = {}
    for q in questions:
        by_table.setdefault(q.table_name, []).append(q)

    req_ids = {r.id for r in requirements}
    formatted = _format_requirements(requirements)
    # The scenario is where the answers to the Requirements table actually live.
    # Extraction turns prose into "The system shall ..." statements and discards
    # the project framing: bank_migration says "No new functionality is to be
    # added" and "must replicate the existing features exactly", which answer
    # REQ-1 and REQ-3 outright, and neither survives into any requirement.

    out: list[CriterionAnswer] = []

    for table_name, table_qs in by_table.items():
        valid = {q.id for q in table_qs}
        # Project Type & Risk genuinely needs the metadata -- regulatory
        # exposure and integration live there, and it scores 4/5 with it.
        # Requirements does not, and scores worse with it.
        body = (_format_statements(requirements)
                if table_qs[0].table == "requirements" else formatted)
        if scenario:
            body += f"\n\nOriginal problem statement:\n{scenario}"
        result = call_structured(
            system_prompt=INFER_PROMPT,
            user_prompt=(
                f"Requirements:\n\n{body}\n\n"
                f"Answer these questions about the project's "
                f"{table_name.lower()}:\n\n{_format_questions(table_qs)}"
            ),
            response_model=CriterionAnswerSet,
            temperature=0.0,
        )
        scenario_only = {q.id for q in table_qs if q.evidence_from == "scenario"}
        for a in result.answers:
            if a.question_id not in valid:
                continue                      # invented id: cannot be scored
            # A question marked evidence_from: scenario cannot be answered from
            # a requirement, so it goes through the same gate the supplied
            # tables do. PTR-4 (funding) and PTR-5 (product lifespan) were
            # answered by both the 3B and the 8B citing REQ-001 and REQ-002 --
            # real ids, and no more capable of testifying about a budget for
            # being real. Without the scenario there is nothing to check
            # against, so the answer cannot stand either way.
            if a.question_id in scenario_only:
                a.evidence = [e for e in a.evidence if e not in req_ids]
                a = _gate_on_evidence([a], scenario)[0]
                out.append(a)
                continue
            # Keep only real requirement ids, and cap at two. Left uncapped the
            # model cites all of them, which satisfies the schema while carrying
            # no information about what actually decided the answer.
            # Keep requirement ids where cited; otherwise keep the model's own
            # quote. Filtering everything to req_ids discarded scenario
            # evidence, which is where the Requirements-table answers come from.
            ids = [e for e in a.evidence if e in req_ids]
            a.evidence = (ids or a.evidence)[:2]
            out.append(a)
    return out


# ── The evidence gate ──────────────────────────────────────────────────

_WORD = re.compile(r"[a-z0-9']+")


def _normalise(text: str) -> str:
    """Lowercase, straighten quotes, collapse whitespace."""
    text = text.lower().replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"').replace("\u2014", " ")
    return re.sub(r"\s+", " ", text).strip()


def _quotes_the_source(quote: str, source: str) -> bool:
    """Is this quote actually traceable to the problem statement?

    Substring after normalisation, else a word-overlap fallback -- the model
    paraphrases as often as it quotes ("the original problem statement says the
    product team is small"), and rejecting a faithful paraphrase would push a
    correct answer to not_stated for a formatting reason.

    This catches invented quotes. It cannot catch a real quote attached to the
    wrong question -- credit_risk answered USER-2 ("are the user representatives
    new to system definition?") YES and cited "they've never built ML systems
    before", which is a true sentence about the TEAM. That needs a reader.
    """
    q, src = _normalise(quote), _normalise(source)
    if not q:
        return False
    if q in src:
        return True
    words = [w for w in _WORD.findall(q) if len(w) > 3]
    if not words:
        return False
    hits = sum(1 for w in words if w in src)
    return hits / len(words) >= 0.6


def _gate_on_evidence(answers: list[CriterionAnswer],
                      source: str) -> list[CriterionAnswer]:
    """Demote to not_stated any yes/no the model could not point at.

    The prompt already says "if you cannot quote them, the answer is
    not_stated". Across five scenarios it was ignored in one direction only:
    every supplied-table answer that got through was YES, and five of the six
    cited nothing at all. USER-2 came back YES in four of five runs -- a
    constant, not an inference -- and USER-2=YES scores for Prototype, Spiral
    and Incremental and for nobody else, so the fabrication was not even
    neutral in its effect on the ranking.

    Enforcing it here rather than asking again is the same trade the extraction
    scope fix and `_format_statements` already make: a rule that can be checked
    in code should not be delegated to the model's compliance with a negative
    instruction.
    """
    out = []
    for a in answers:
        if a.as_bool is not None:
            kept = [e for e in a.evidence if _quotes_the_source(e, source)]
            if not kept:
                claimed = a.answer
                a.answer = "not_stated"
                a.evidence = []
                a.reasoning = (f"UNSUPPORTED (claimed {claimed}, nothing "
                               f"traceable to the problem statement): "
                               f"{a.reasoning}")
            else:
                a.evidence = kept[:2]
        out.append(a)
    return out


CONTEXT_PROMPT = """\
You are a software engineering analyst. Given a project's problem statement, \
answer questions about how the PROJECT will be run — its team and its users.

Two failures are possible here and they are NOT symmetric in how often they \
happen. Inventing an answer is the obvious one. The commoner one is missing an \
answer that is sitting in the text in different words.

Rules:
- Answer ONLY from the problem statement. Never infer a team's experience from \
the kind of system being built, or users' availability from the fact that they \
are users.
- **But when the statement DOES say something, record it — even if it does not \
use the words in the question.** These questions are phrased for a scoring \
sheet; a real brief never phrases things that way. "The company has \
considerable prior experience in product lines similar to this" is a direct \
answer to whether the team is new to the problem domain. So is "they have never \
built ML systems before", in the other direction.
- `not_stated` is for subjects the statement does not touch at all — not for \
subjects it addresses in its own words.
- When you answer yes or no, quote the words that say so in `evidence`. If you \
cannot quote them, the answer is not_stated.
- `reasoning` is one sentence naming what decided it.
- Answer EVERY question, using the exact question ids.

Worked examples, in both directions:

  "...they have never built ML systems before."
  Q: Are the majority of team members new to the technology domain?
  → yes. Evidence "never built ML systems before".

  "The company has considerable prior experience in product lines similar to
   this."
  Q: Are the majority of team members new to the problem domain?
  → NO. Prior experience in similar products is the opposite of new to it.
    Evidence "considerable prior experience in product lines similar to this".

  "...then iterate based on user feedback."
  Q: Do the users want to be involved in all phases of the life cycle?
  → yes. Evidence "iterate based on user feedback".

  "The team has 15 developers, funding is approved for 18 months."
  Q: Is there training available for the project team if required?
  → not_stated. Team SIZE and FUNDING are given; training is not mentioned at
    all, in any words.

Both outcomes are common. Declining a question the statement does not address
is as important as answering one it does — and answering one it does address is
as important as declining one it does not.
"""


# Vocabulary that marks a sentence as being ABOUT the team or ABOUT the users.
# Deliberately generous -- a false positive costs one sentence of prompt, a
# false negative loses a fact the scenario actually states.
_MENTIONS = {
    "project_team": re.compile(
        r"\b(team|teams|developer|developers|engineer|engineers|programmer|"
        r"staff|staffing|headcount|people|resource|resources|hire|hired|"
        r"hiring|experience|experienced|expertise|skilled|skill|familiar|"
        r"trained|training|novice|never built|first time|in-house|contractor|"
        r"vendor|reassign|replaced|dedicated)\b", re.I),
    "user_community": re.compile(
        r"\b(user|users|customer|customers|client|clients|stakeholder|"
        r"stakeholders|representative|representatives|end.user|audience|"
        r"feedback|involve|involved|involvement|consulted|available|"
        r"availability|business user|domain expert)\b", re.I),
}

_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


def relevant_sentences(scenario: str, table: str) -> list[str]:
    """The sentences that say anything at all about this table's subject.

    A deterministic pre-filter, and the fix for the thing that made this step
    oscillate. One LLM call was answering two different questions at once --
    "does the scenario mention the team?" and "what does it say?" -- and the
    first is retrieval, which a regex does perfectly, while the second is
    judgement. Conflated, the model would either read the team out of a brief
    that never mentions one, or miss "considerable prior experience in product
    lines similar to this" because the surrounding text was about wristwatches.

    Separated: if nothing here matches, the scenario genuinely does not discuss
    the subject, no model is asked to pretend otherwise, and the interview asks.
    If something matches, the model sees only those sentences -- a much smaller
    and more pointed task than reading a whole brief for nine rubric questions.
    """
    pattern = _MENTIONS.get(table)
    if not pattern:
        return [scenario] if scenario.strip() else []
    return [s.strip() for s in _SENTENCE.split(scenario)
            if s.strip() and pattern.search(s)]


def infer_project_context(scenario: str,
                          questions: list[Question]) -> list[CriterionAnswer]:
    """Answer the Project Team and User Community tables from the scenario.

    These were previously unassessed on the grounds that a problem statement
    says nothing about team size. That was true of the original scenarios and is
    not true of the current ones -- "the project team consists of 2 developers",
    "the team has 15 developers, funding is approved for 18 months", "the product
    team is small, 3 people". The facts are there; nothing was reading them.

    Answering them roughly doubles the number of scored questions, which matters
    beyond coverage: on ten binary questions two models tied at 70% twice across
    five scenarios, and an alphabetical tie-break picked the winner.
    """
    if not scenario.strip():
        return []

    by_table: dict[str, list[Question]] = {}
    for q in questions:
        by_table.setdefault(q.table, []).append(q)

    out: list[CriterionAnswer] = []
    for table, table_qs in by_table.items():
        table_name = table_qs[0].table_name
        valid = {q.id for q in table_qs}

        # Does the scenario discuss this subject at all? Decided in code, not by
        # the model. When it does not, no call is made -- there is nothing to
        # read, and asking a model to read it anyway is how the fabricated team
        # answers started.
        sentences = relevant_sentences(scenario, table)
        if not sentences:
            out.extend(CriterionAnswer(
                question_id=q.id, answer="not_stated", evidence=[],
                reasoning=f"The problem statement does not mention the "
                          f"{table_name.lower()} at all.")
                for q in table_qs)
            continue

        # It does -- so the model reads ONLY those lines. A pointed task on two
        # sentences beats the same task on a whole brief, and it cannot drift
        # onto text that has nothing to do with the question.
        excerpt = "\n".join(f"- {x}" for x in sentences)
        result = call_structured(
            system_prompt=CONTEXT_PROMPT,
            user_prompt=(
                f"These are the only lines in the problem statement that "
                f"mention the {table_name.lower()}:\n\n{excerpt}\n\n"
                f"Answer these questions from them:\n\n"
                f"{_format_questions(table_qs)}"
            ),
            response_model=CriterionAnswerSet,
            temperature=0.0,
        )
        out.extend(_gate_on_evidence(
            [a for a in result.answers if a.question_id in valid],
            "\n".join(sentences)))
    return out


REPLY_PROMPT = """\
A person was just asked about their project and this is their answer. Read it \
and record what they said.

This is a REPLY, not a document. They were asked these exact questions and are \
answering them, so **expect answers, not silence.** Reading a direct answer as \
"not stated" throws away the thing they took the trouble to tell you.

Rules:
- **A hedge is an answer.** "maybe they're new to the tech", "I think so", \
"probably not" all answer the question. Record the direction they lean. Only \
`not_stated` when they did not touch the subject at all.
- Informal spelling and grammar are normal here. Read what they meant: "knew to \
technology" means NEW to the technology; "ya" means yes; "2-3 member" means a \
team of two or three.
- One sentence often answers several questions. "they know the domain but are \
new to the tech and training is available" answers three.
- **But do not stretch one sentence onto a question it does not address.** The \
business area, the technology and the specific tools are three different \
questions: being new to the technology says nothing about the tools, and a \
quote about one is not evidence about another. If they did not mention it, it \
is `not_stated`.
- Read the whole reply for each question before answering. "they are not to be \
replaced" answers whether people will be reassigned, even though it does not \
use that word.
- Quote their own words in `evidence`. If you cannot quote them, it is \
`not_stated`.
- `reasoning` is one sentence naming what they said that decided it.
- Answer EVERY question, using the exact question ids.
"""


def infer_from_reply(reply: str,
                     questions: list[Question]) -> list[CriterionAnswer]:
    """Read framework answers out of what a person said when asked.

    Separate from `infer_project_context` because the two read different things
    and need opposite defaults. A problem statement is usually silent on the
    team, so that prompt leans hard on `not_stated` -- correctly, since inferring
    there produced fabrications in every run. A reply to a direct question is the
    opposite: silence is the unlikely case, and the same caution made the model
    answer "not stated" to "training is available" when asked whether training
    was available.

    The evidence gate is unchanged. An answer that cannot be quoted from what
    they actually said is still dropped, so a generous reading buys no licence
    to invent.
    """
    if not reply.strip():
        return []

    by_table: dict[str, list[Question]] = {}
    for q in questions:
        by_table.setdefault(q.table_name, []).append(q)

    out: list[CriterionAnswer] = []
    for table_name, table_qs in by_table.items():
        valid = {q.id for q in table_qs}
        result = call_structured(
            system_prompt=REPLY_PROMPT,
            user_prompt=(f"They were asked about the project's "
                         f"{table_name.lower()} and said:\n\n\"{reply}\"\n\n"
                         f"Record their answers to:\n\n"
                         f"{_format_questions(table_qs)}"),
            response_model=CriterionAnswerSet,
            temperature=0.0,
        )
        out.extend(_gate_on_evidence(
            [a for a in result.answers if a.question_id in valid], reply))
    return out


# ── 2. Cross-check: the model's own ranking, ungrounded in the scores ──

CROSS_CHECK_PROMPT = """\
You are a software engineering analyst. Given a set of software requirements for \
a financial system, rank these SDLC models from most to least suitable:

{models}

Use your own judgement about the requirements. Give two sentences on why the top \
choice fits. Use the model names exactly as written above.
"""


def cross_check(requirements: list[Requirement], framework: Framework) -> CrossCheckRanking:
    return call_structured(
        system_prompt=CROSS_CHECK_PROMPT.format(
            models="\n".join(f"- {m.name}" for m in framework.models)
        ),
        user_prompt=f"Requirements:\n\n{_format_requirements(requirements)}",
        response_model=CrossCheckRanking,
        temperature=0.0,
    )


# ── 3. Justify the ranking the framework produced ──────────────────────

JUSTIFY_PROMPT = """\
You are a software engineering analyst writing the rationale for an SDLC \
selection that has ALREADY been made by scoring the project against a selection \
framework. Explain the outcome — do not re-decide it.

You are given, for each candidate, every framework question its profile matched \
and the ones it did not, with BOTH answers and a MATCH/MISMATCH verdict.

Fill the fields:

- `advantages` — each candidate's advantages are LISTED FOR YOU under its \
scoring block. Copy each `advantage` across **exactly as written** and fill in \
only `relevance`: one sentence on what that property buys THIS project, \
pointing at something concrete from the scenario or the requirements. Quote the \
requirement rather than citing "REQ-002" alone. A `relevance` line that would \
read the same for any financial system is not an answer, and the three \
candidates must not share one.
- `shortcut` — one line for the whole decision: three to five traits **of this \
project**, joined by "+", then "->" and the chosen model. Take every trait from \
the FITS list you were given. Do not reuse wording from any example; a shortcut \
naming a system this project is not is worse than no shortcut.

Rules:
- The FINAL STANDING is given and is already decided. Do not re-rank, and name \
no runner-up other than the one stated.
- Do not count. Each scoreline is given; never state a total that contradicts it.
- **No question ids anywhere in your output.** They are internal labels for a \
scoring sheet; a reader sees only the criterion itself. Requirement ids are \
different — REQ-002 is a label for something printed above, so it may be cited, \
but quote the requirement alongside it.
"""


def justify(requirements: list[Requirement], framework: Framework,
            top, answers: list[CriterionAnswer],
            tied_with: list[str] | None = None) -> JustificationSet:
    by_id = {a.question_id: a for a in answers}

    def render(qid: str, model_id: str, model_name: str) -> str:
        """One question, with BOTH answers and the direction spelled out.

        Two bugs have come from under-specifying this line, and they are the
        same bug twice.

        First the project's answer was missing, so a matched question read as a
        "yes": one run justified Incremental with "the requirements are
        well-defined (REQ-1)" when REQ-1 was answered NO and Incremental matched
        *because* it expects no.

        Then the MODEL's expected answer was still missing, so the direction of
        a mismatch had to be inferred -- and it was inferred backwards. Shown
        only "Waterfall disagrees on PTR-6, project answered YES", the model
        concluded Waterfall "is suitable for ... a need for thorough testing at
        every stage", and in the same run said V-Shaped "does not fully account
        for the project's need for formal verification and testing" -- the exact
        inverse, on the one question that exists to separate them.

        So nothing is left to inference: both answers appear, and the verdict is
        stated in words rather than implied by which list the line is in.
        """
        q = framework.question(qid)
        a = by_id.get(qid)
        given = (a.answer if a else "no").upper()
        expects = "YES" if q.expects[model_id] else "NO"
        agrees = given == expects
        verdict = (f"MATCH — {model_name} expects this"
                   if agrees else
                   f"MISMATCH — {model_name} assumes the OPPOSITE")
        return (f'{qid} "{q.text}"\n'
                f'        project: {given}   |   {model_name} expects: {expects}'
                f'   ->  {verdict}')

    blocks = []
    for rank, s in enumerate(top, start=1):
        mid, mname = s.model.id, s.model.name
        # Every match, capped misses. The old 6/4 split gave the WINNER a third
        # of its evidence as reasons against itself, and the justification read
        # like a case for the prosecution. Worse, truncating matches made the
        # model fill the gap: with 8 matches cut to 6 it asserted Prototype
        # "assumes the funding will be stable, which is not the case" -- PTR-4
        # was a match that fell off the end.
        matched = "\n    ".join(render(q, mid, mname) for q in s.matched) or "none"
        missed = "\n    ".join(render(q, mid, mname) for q in s.missed[:3]) or "none"
        # A scoreline the model cannot miscount. It repeatedly invented totals
        # ("matches on 6 out of 11 properties" for a model scoring 10/11).
        role = ("RECOMMENDED — explain why it WON"
                if rank == 1 else f"ranked #{rank}")
        adv = "\n    ".join(f"- {a}" for a in s.model.advantages) or "none listed"
        blocks.append(
            f"{mname} [{role}] — {len(s.matched)} of {s.answered} match "
            f"({s.score:.0%})\n"
            f"  {mname.upper()}'S ADVANTAGES (copy these verbatim, write only "
            f"the relevance):\n    {adv}\n"
            f"  FITS on {len(s.matched)}:\n    {matched}\n"
            f"  DOES NOT FIT on {len(s.missed)}"
            + (f" (showing 3)" if len(s.missed) > 3 else "")
            + f":\n    {missed}"
        )

    evidence = "\n".join(
        f"{a.question_id} = {a.answer} "
        f"[{', '.join(a.evidence) or 'no requirement cited'}] {a.reasoning}"
        for a in answers
    )
    # Name the runner-up rather than leaving it to be worked out. The summary
    # field asks what "the runner-up" would need to win, and nothing told the
    # model which one that was -- it inferred it from the blocks and could as
    # easily have named #3, or a model tied at a lower score. The ranking is
    # already computed; stating it costs one line and removes the guess.
    standing = "\n".join(
        f"  #{i}  {x.model.name}  {x.score:.0%} ({len(x.matched)}/{x.answered})"
        f"{'   <- RECOMMENDED' if i == 1 else ''}"
        f"{'   <- RUNNER-UP' if i == 2 else ''}"
        for i, x in enumerate(top, start=1))
    runner_up = top[1].model.name if len(top) > 1 else None
    standing_note = (
        f"\n\nFINAL STANDING (already decided — do not re-rank):\n{standing}\n"
        + (f'When the summary says "the runner-up", it means {runner_up}. '
           f"Name no other model as the runner-up.\n" if runner_up else "")
    )

    tie_note = ""
    if tied_with and len(tied_with) > 1:
        # Without this the summary invents a margin: asked to name "the
        # runner-up and what would have to change for it to win", a model shown
        # two identical scores writes a reason the first one is better anyway.
        tie_note = (
            f"\n\nIMPORTANT: {' and '.join(tied_with)} are TIED on the "
            f"framework -- identical scores, no separation. Say so plainly, and "
            f"say which framework question would break the tie if it were "
            f"answered. Do not claim one is the better fit.\n"
        )

    result = call_structured(
        system_prompt=JUSTIFY_PROMPT,
        user_prompt=(
            f"Requirements:\n\n{_format_requirements(requirements)}\n\n"
            f"How the project answered the framework:\n{evidence}\n\n"
            f"Scoring outcome:\n\n" + "\n\n".join(blocks)
            + standing_note + tie_note
        ),
        response_model=JustificationSet,
        temperature=0.0,
    )

    # Fill in the fits from the scoring, not from the model. Every bullet here
    # is a question this model demonstrably matched, phrased from the framework
    # file -- so it cannot list a mismatch as a reason, cannot invert a
    # direction, and cannot miscount.
    by_name = {s.model.name: s for s in top}
    for j in result.justifications:
        scored = by_name.get(j.model)
        if not scored:
            continue
        # The advantage names come from the framework, not the model -- it
        # inverted the two fields (a whole sentence in `advantage`, a bare
        # "REQ-002" in `relevance`) and emitted one identical list for all three
        # candidates. Only `relevance` is its work, so only `relevance` is kept.
        canon = list(scored.model.advantages)
        if canon:
            paired = []
            for i, a in enumerate(j.advantages[:len(canon)]):
                a.advantage = canon[i]
                paired.append(a)
            j.advantages = paired
        # The cost, composed from what this model actually missed. Generated,
        # it claimed V-Shaped "gives up the need for formal verification and
        # testing at multiple stages" -- which is V-Shaped's defining property.
        # A mismatch means the project needs something this model does not
        # provide, and that is arithmetic, not judgement.
        # Phrased from the MODEL's assumption, not the project's answer. A miss
        # means the methodology is built for a situation this project is not in,
        # so "X assumes <what X expects>, which is not the case here" is both
        # correct and readable. Framing it the other way round produced "This
        # project needs this is not an extension of an existing system".
        costs = []
        for qid in scored.missed[:2]:
            q = framework.question(qid)
            if q:
                costs.append(q.says(q.expects[scored.model.id]))
        if costs:
            body = costs[0] + (f", and that {costs[1]}" if len(costs) > 1 else "")
            tail = ("neither is true here" if len(costs) > 1 else "that is not the "
                    "case here")
            j.gives_up = (f"{j.model} assumes {body} — {tail}, so that part of "
                          f"its design earns nothing on this project.")
        else:
            j.gives_up = (f"Nothing in this project's profile works against "
                          f"{j.model}.")

        j.why_it_fits = [
            framework.question(qid).says(by_id[qid].as_bool)
            for qid in scored.matched
            if framework.question(qid) and qid in by_id
            and by_id[qid].as_bool is not None
        ][:6]
    return result


# ── Orchestration ──────────────────────────────────────────────────────

def recommend_sdlc(requirements: list[Requirement],
                   framework: Framework | None = None,
                   supplied_answers: dict[str, bool] | None = None,
                   run_cross_check: bool = True,
                   scenario: str = "",
                   ask_user=None) -> SDLCRecommendation:
    """Requirements in, ranked SDLC recommendation with justification out.

    `supplied_answers` covers the Project Team and User Community tables, which
    cannot be inferred from a problem scenario -- a 52-word brief says nothing
    about team experience. Absent, those tables are simply not scored, and
    `coverage` reports them as unassessed rather than letting a half-scored
    result pass as a full framework assessment.
    """
    fw = framework or Framework.load()

    inferred = infer_criteria(requirements, fw.questions_for("inferred"), scenario)
    # The supplied tables are read from the scenario rather than asked for. A
    # question the scenario does not address comes back not_stated and is left
    # unscored, so coverage reports what was actually assessed.
    inferred += infer_project_context(scenario, fw.questions_for("supplied"))

    answer_map: dict[str, bool] = {
        a.question_id: a.as_bool for a in inferred if a.as_bool is not None
    }
    if supplied_answers:
        answer_map.update({k: bool(v) for k, v in supplied_answers.items()
                           if fw.question(k)})

    # Fill the gaps BEFORE scoring, not after. `ask_user` is supplied only when
    # a person is at a terminal; it receives the answers so far and returns any
    # it could establish by asking. Doing this after scoring would mean showing
    # a recommendation and then retracting it, which teaches a reader to
    # distrust the first number -- and the first number was never meant to be
    # shown.
    if ask_user is not None:
        # An answer with no evidence behind it is a guess that scores exactly
        # like a fact. Offer those for asking about too, not just the blanks.
        weak = {a.question_id for a in inferred
                if a.as_bool is not None and not a.evidence}
        extra = ask_user(fw, dict(answer_map), weak) or {}
        for qid, val in extra.items():
            if fw.question(qid):
                answer_map[qid] = bool(val)
                # Record it alongside the inferred answers so `criteria` and
                # `coverage` describe the decision that was actually made.
                inferred = [a for a in inferred if a.question_id != qid]
                inferred.append(CriterionAnswer(
                    question_id=qid, answer="yes" if val else "no",
                    evidence=["supplied by the user when asked"],
                    reasoning="Established in conversation, not from the scenario."))

    scores = fw.score(answer_map)
    top = scores[:TOP_N]
    # Only the winner is explained. Justifying the runners-up produced three
    # near-identical blocks -- the same fits, the same advantages, the same
    # cost -- because the interesting differences are small and the model
    # padded. The ranking is still printed; what a reader wants is one correct
    # argument, not three interchangeable ones.
    explained = scores[:1]
    tied = [s.model.name for s in scores if abs(s.score - scores[0].score) < 1e-9]
    tied_with = tied if len(tied) > 1 else []

    xc = cross_check(requirements, fw) if run_cross_check else None
    just = justify(requirements, fw, explained, inferred, tied_with)

    agrees = None
    if xc and xc.ranked_models:
        agrees = xc.ranked_models[0].strip().lower() == top[0].model.name.lower()

    return SDLCRecommendation(
        recommendation=top[0].model.name,
        tied_with=tied_with,
        ranking=[ScoredModel(model_id=s.model.id, name=s.model.name,
                             score=round(s.score, 4), matched=s.matched,
                             missed=s.missed, is_derived=s.model.is_derived)
                 for s in scores],
        criteria=inferred,
        justifications=just.justifications,
        summary=just.recommendation_summary,
        shortcut=just.shortcut,
        coverage={k: TableCoverage(**v) for k, v in fw.coverage(answer_map).items()},
        cross_check_ranking=xc.ranked_models if xc else [],
        cross_check_reasoning=xc.reasoning if xc else "",
        cross_check_agrees=agrees,
        framework_version=fw.version,
        uses_derived_models=any(s.model.is_derived for s in top),
    )
