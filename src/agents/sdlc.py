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
    return "\n".join(f"{q.id}: {q.text}" for q in questions)


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
        for a in result.answers:
            if a.question_id not in valid:
                continue                      # invented id: cannot be scored
            # Keep only real requirement ids, and cap at two. Left uncapped the
            # model cites all of them, which satisfies the schema while carrying
            # no information about what actually decided the answer.
            a.evidence = [e for e in a.evidence if e in req_ids][:2]
            out.append(a)
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
selection that has already been made by scoring the project against a selection \
framework.

You are given, for each of the top candidates, which framework questions its \
profile matched and which it did not. Explain the outcome — do not re-decide it.

Rules:
- Cite framework question ids (REQ-3, PTR-2, ...) and requirement ids (REQ-002) \
in every justification.
- Each question shows the answer the PROJECT gave. A model "agrees" when its \
expected answer equals the project's — which often means agreeing on a NO. \
Never describe a matched question as though the project answered yes; read the \
stated answer.
- Say what each model gives up, not only what it offers.
- In the summary, name the runner-up and state what would have to be different \
about the project for it to win instead.
- Be concrete about this project. A justification that would read the same for \
any financial system is not useful.
"""


def justify(requirements: list[Requirement], framework: Framework,
            top, answers: list[CriterionAnswer]) -> JustificationSet:
    by_id = {a.question_id: a for a in answers}

    def render(qid: str) -> str:
        # The project's answer has to appear beside the question. Without it the
        # model reads a matched question as a "yes": the first run justified
        # Incremental with "the requirements are well-defined (REQ-1)" when
        # REQ-1 was answered NO and Incremental matched *because* it expects no.
        q = framework.question(qid)
        a = by_id.get(qid)
        given = "yes" if (a and a.answer) else "no"
        return f'{qid} "{q.text}" -> project answered {given.upper()}'

    blocks = []
    for s in top:
        matched = "\n    ".join(render(q) for q in s.matched[:6]) or "none"
        missed = "\n    ".join(render(q) for q in s.missed[:4]) or "none"
        blocks.append(
            f"{s.model.name} — scored {s.score:.0%} ({len(s.matched)}/{s.answered})\n"
            f"  AGREES with this model on:\n    {matched}\n"
            f"  DISAGREES on:\n    {missed}"
        )

    evidence = "\n".join(
        f"{a.question_id} = {'yes' if a.answer else 'no'} "
        f"[{', '.join(a.evidence) or 'no requirement cited'}] {a.reasoning}"
        for a in answers
    )
    return call_structured(
        system_prompt=JUSTIFY_PROMPT,
        user_prompt=(
            f"Requirements:\n\n{_format_requirements(requirements)}\n\n"
            f"How the project answered the framework:\n{evidence}\n\n"
            f"Scoring outcome:\n\n" + "\n\n".join(blocks)
        ),
        response_model=JustificationSet,
        temperature=0.0,
    )


# ── Orchestration ──────────────────────────────────────────────────────

def recommend_sdlc(requirements: list[Requirement],
                   framework: Framework | None = None,
                   supplied_answers: dict[str, bool] | None = None,
                   run_cross_check: bool = True,
                   scenario: str = "") -> SDLCRecommendation:
    """Requirements in, ranked SDLC recommendation with justification out.

    `supplied_answers` covers the Project Team and User Community tables, which
    cannot be inferred from a problem scenario -- a 52-word brief says nothing
    about team experience. Absent, those tables are simply not scored, and
    `coverage` reports them as unassessed rather than letting a half-scored
    result pass as a full framework assessment.
    """
    fw = framework or Framework.load()

    inferred = infer_criteria(requirements, fw.questions_for("inferred"), scenario)
    answer_map: dict[str, bool] = {a.question_id: a.answer for a in inferred}
    if supplied_answers:
        answer_map.update({k: bool(v) for k, v in supplied_answers.items()
                           if fw.question(k)})

    scores = fw.score(answer_map)
    top = scores[:TOP_N]

    xc = cross_check(requirements, fw) if run_cross_check else None
    just = justify(requirements, fw, top, inferred)

    agrees = None
    if xc and xc.ranked_models:
        agrees = xc.ranked_models[0].strip().lower() == top[0].model.name.lower()

    return SDLCRecommendation(
        recommendation=top[0].model.name,
        ranking=[ScoredModel(model_id=s.model.id, name=s.model.name,
                             score=round(s.score, 4), matched=s.matched,
                             missed=s.missed, is_derived=s.model.is_derived)
                 for s in scores],
        criteria=inferred,
        justifications=just.justifications,
        summary=just.recommendation_summary,
        coverage={k: TableCoverage(**v) for k, v in fw.coverage(answer_map).items()},
        cross_check_ranking=xc.ranked_models if xc else [],
        cross_check_reasoning=xc.reasoning if xc else "",
        cross_check_agrees=agrees,
        framework_version=fw.version,
        uses_derived_models=any(s.model.is_derived for s in top),
    )
