"""Quality analysis agent — checks requirements for ambiguity, incompleteness,
inconsistency, and other quality issues.

Day 6: LLM-based quality checker. Sends ALL requirements in one call so
cross-requirement issues (duplication, inconsistency) can be detected.
"""

import re
from functools import lru_cache
from pathlib import Path

import yaml

from src.models.quality import QualityReport
from src.models.requirement import Requirement
from src.utils.llm import call_structured

SYSTEM_PROMPT = """\
You are a senior QA engineer reviewing software requirements for a financial \
system. Analyse EACH requirement for the following quality issues:

1. **Ambiguity** — Vague or unmeasurable terms like "fast", "quickly", \
"user-friendly", "efficient", "appropriate", "etc.". Every requirement must \
have a concrete, measurable definition.
2. **Incompleteness** — Missing critical details that the requirement itself \
should state: who performs the action, what triggers it, what the expected \
outcome is, or missing boundary values. **Unstated failure handling is NOT \
incompleteness.** "What happens if it fails?", "what if the service is down?", \
"what about the error case?" are design questions, not defects in the \
requirement — do not raise them.
3. **Inconsistency** — Contradicts another requirement in the list (e.g. \
one says "5 seconds" and another says "3 seconds" for the same operation).
4. **Duplication** — Says the same thing as another requirement using \
different wording.
5. **Infeasibility** — Technically impossible or unrealistic given current \
technology or reasonable cost constraints.
6. **Lack of Testability** — Cannot be objectively verified with a test \
case. Requirements like "shall be intuitive" fail this check unless they \
reference a measurable standard (e.g. WCAG 2.1 AA).
7. **Missing Source** — No stakeholder or source identified for the \
requirement (check the source_stakeholder field).
8. **Undefined Terminology** — Uses a term **this system coins for itself** \
without saying what it means: "senior officer", "high-risk customer", \
"the standard fee", "tier-2 merchant". Ask whether a reader could look the term \
up in a published source. If they could — UPI, eKYC, PCI-DSS, PEP, Aadhaar, \
ISO 20022 are all defined by a regulator or a standards body — it is NOT \
undefined, and demanding a definition is asking the requirement to restate a \
regulation. Named regulations and document titles are references, not jargon.
9. **Conflicting Expectations** — Two different stakeholders want opposite \
behaviours or contradictory priorities.
10. **Missing security/compliance** — The requirement handles sensitive data \
(PII, financial data, biometrics) but does not mention encryption, access \
control, or applicable regulations.

Rules:
- Check EVERY requirement in the list — do not skip any.
- **Read every field of a requirement before flagging it.** If the acceptance \
criteria already give a measurable condition, the requirement is testable. If \
the assumptions already state a precondition, it is not missing. Do not ask \
for information that is already there — that is the most common mistake on \
this task.
- If a requirement passes ALL 10 checks, set passed=True and return an \
empty issues list for that requirement. **Requirements that are genuinely \
fine are common. Passing them is a correct answer, not a failure to find \
something.**
- If a requirement has problems, set passed=False and list each issue with \
a specific, actionable suggestion for how to fix it.
- For cross-requirement issues (Inconsistency, Duplication, Conflicting \
Expectations), reference the conflicting requirement IDs — and apply a strict \
test before you do. **Two requirements conflict only if they constrain the \
SAME thing to values that cannot both hold** (e.g. one says a response must \
be under 3 seconds, another under 5 seconds, for the same operation). \
Requirements about different subjects never contradict each other, however \
different they sound: escalating high-risk customers does not contradict \
retaining recordings for 8 years, and a rule about low-risk customers does not \
contradict a rule about high-risk ones. If you cannot name the shared thing \
and the two incompatible values, there is no conflict — do not report one.
- Severity discipline: **Critical/High are for issues that would cause \
incorrect behaviour, a compliance breach, or rework** — a genuine \
contradiction, a missing security control on sensitive data. Vague wording and \
undefined acronyms are Low or Medium. Do not inflate severity.
- Be strict but fair — flag real problems, not nitpicks.
- **Quote the defect.** Every issue description must quote, in double quotes, \
the exact words from the requirement that are wrong, vague or missing a \
definition — for example: the term "user-friendly" is not measurable. **If you \
cannot point at specific words that are defective, there is no issue and you \
must not report one.** A complaint that would read the same against any \
requirement is not a finding.
- Before returning, re-read each issue and delete any that (a) quotes nothing \
from the requirement, (b) asks for error or failure handling, or (c) asks for \
something another field of that requirement already states.
"""


# Rules the model will not follow, enforced here instead.
#
# Three prompt revisions asked llama3.2:3b not to raise unstated failure
# handling as a defect -- naming it in the check definition, in the rules, and
# in a self-review step. It raised it on five of six requirements every time.
# A 3B model does not reliably obey a negative constraint, and a rule you can
# check deterministically should not be delegated to one. Same reasoning as the
# JSON salvage layer in llm_ollama: ask for good output, then enforce what
# matters.
#
# Kept narrow on purpose. This drops issues that are *categorically* invalid
# under the rubric, never issues that are merely weak -- judging weak ones is
# the model's job, and a filter that second-guesses it would hide real findings.
REPO = Path(__file__).resolve().parents[2]

_FAILURE_HANDLING = re.compile(
    r"what happens if|if .{0,40}\b(fails?|failed|is not|are not|does not|cannot)\b"
    r"|failure (case|mode|scenario|handling)|error (case|handling)",
    re.I,
)

# A citation is not undefined jargon. The agent flagged "RBI/DBR/2015-16/18" as
# an undefined term; it is a document reference, and one the grounding node put
# there on purpose.
_LOOKS_LIKE_CITATION = re.compile(r"[A-Z]{2,}[/\-]\d|§|\bNo\.\s*\d|\b\d{4}-\d{2}\b")


# Inconsistency, Duplication and Conflicting Expectations are by definition
# relationships BETWEEN requirements -- each needs a second requirement to be
# inconsistent with. The model applies them to single-requirement wording
# problems instead: "mobile-responsive is not defined" came back labelled
# Inconsistency in three runs out of three. The finding is real; only the label
# is wrong, so relabel rather than discard.
_CROSS_REQUIREMENT = {"Inconsistency", "Duplication", "Conflicting Expectations"}
_REQ_REF = re.compile(r"\bREQ[-\s]?\d+", re.I)


def _relabel(issue):
    if issue.issue_type in _CROSS_REQUIREMENT and not _REQ_REF.search(issue.description):
        # No second requirement named, so this cannot be a relationship defect.
        # Ambiguity is the honest fallback: it is what a complaint about
        # undefined or vague wording actually is.
        issue.issue_type = "Ambiguity"
    return issue


def _keep(issue) -> bool:
    if issue.issue_type == "Incompleteness" and _FAILURE_HANDLING.search(issue.description):
        return False
    if issue.issue_type == "Undefined Terminology" and _LOOKS_LIKE_CITATION.search(issue.description):
        return False
    return True


@lru_cache(maxsize=1)
def _established_terms() -> tuple[str, ...]:
    """Terms with a published, authoritative definition. See the YAML for why.

    Loaded from data rather than hard-coded so each entry carries its
    justification and a reader can audit the list. Missing file is not fatal --
    the check simply stops firing, which degrades to the old behaviour rather
    than crashing a run.
    """
    path = REPO / "data/quality/established_terms.yaml"
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return tuple(t["term"].lower() for t in data.get("terms", []))
    except Exception:
        return ()


_NAMED_DOCUMENT = re.compile(
    r"\b(guidelines?|regulations?|directions?|act|framework|standards?|"
    r"master\s+direction|circular|notification)\b", re.I)


def _is_established_term(issue) -> bool:
    """Is the flagged term something a reader can simply look up?

    Two ways to qualify. Either the quoted term contains a glossary entry
    ("Aadhaar-based eKYC" contains "eKYC", "PCI-DSS standards" contains
    "PCI-DSS"), or it names a document rather than a concept ("RBI's digital
    lending guidelines" is a reference, not undefined jargon).
    """
    if issue.issue_type != "Undefined Terminology":
        return False
    quoted = re.findall(r"['\"]([^'\"]{2,70})['\"]", issue.description)
    if not quoted:
        return False
    terms = _established_terms()
    for q in quoted:
        low = q.lower()
        if any(re.search(rf"\b{re.escape(t)}\b", low) for t in terms):
            return True
        if _NAMED_DOCUMENT.search(q):
            return True
    return False


_NUMERIC = re.compile(r"\d")

# Vague words a NUMBER could actually pin down. The distinction matters: "the
# system shall respond in real-time" is answered by "within 2 seconds", but
# "approval from a senior officer" is not answered by any number -- and its
# acceptance criteria happen to say "above 50 lakhs", which quantifies the
# threshold, not the role. Without this list that unrelated number silenced a
# correct finding.
_MEASURABLE_VAGUENESS = re.compile(
    r"\b(real[\s-]?time|responsive|fast|quick|rapid|timely|immediate|prompt|"
    r"instant|low[\s-]?latency|high[\s-]?performance|scalable|concurrent|"
    r"frequent|minimal|large|small|accessible|available)\b", re.I)


def _quantified_in_acceptance(issue, requirement) -> bool:
    """Already measurable, so the complaint is answered by the requirement itself.

    The prompt says "if the acceptance criteria already give a measurable
    condition, the requirement is testable" -- and the model ignores it. REQ-003
    was flagged for "real-time" while its own acceptance criteria said "within 2
    seconds"; REQ-001 for "mobile-responsive" while its acceptance said "screen
    widths from 320px to 1920px".

    Three conditions, all required, because the first draft of this was too
    loose and silenced a correct complaint about "senior officer":

      1. the issue is about wording, not about a missing control
      2. the flagged term is one a NUMBER could pin down -- a dimension, not a
         role or an entity
      3. the acceptance criteria contain a number

    "Real-time" with no latency target anywhere still fails, and stays.
    """
    if issue.issue_type not in ("Ambiguity", "Lack of Testability",
                                "Undefined Terminology"):
        return False
    criteria = (requirement.acceptance_criteria or "") if requirement else ""
    if not criteria or not _NUMERIC.search(criteria):
        return False
    quoted = re.findall(r"['\"]([^'\"]{2,70})['\"]", issue.description)
    return any(_MEASURABLE_VAGUENESS.search(q) for q in quoted)


def _quotes_a_citation(issue, requirement) -> bool:
    """Is the flagged term part of a citation rather than the requirement's own wording?

    The grounding node writes real citations into applicable_regulations --
    "Master Direction on Outsourcing of Information Technology Services",
    "Digital Personal Data Protection Act, 2023 (Act No. 22 of 2023)". The
    reviewer reads them as undefined jargon and asks for "Master Direction" and
    "Act No." to be defined. They are document titles, correctly attached.

    The pattern check above catches the machine-readable part (§, RBI/2023-24),
    but a complaint quoting only the prose slips through, so compare against the
    citations actually on this requirement.
    """
    if not requirement or not requirement.applicable_regulations:
        return False
    haystack = " ".join(requirement.applicable_regulations).lower()
    for quoted in re.findall(r"['\"]([^'\"]{4,60})['\"]", issue.description):
        if quoted.lower() in haystack:
            return True
    return False


def filter_issues(report: QualityReport,
                  requirements: list[Requirement] | None = None) -> QualityReport:
    """Drop categorically invalid issues, and re-derive `passed` from what is left.

    `passed` has to be recomputed: a requirement whose only complaints were
    filtered out is a requirement that passed, and leaving passed=False would
    report a defect the report no longer contains.
    """
    by_id = {r.id: r for r in (requirements or [])}
    for result in report.result_list:
        req = by_id.get(result.requirement_id)
        result.issues = [
            _relabel(i) for i in result.issues
            if _keep(i)
            and not _quotes_a_citation(i, req)
            and not _is_established_term(i)
            and not _quantified_in_acceptance(i, req)
        ]
        result.passed = not result.issues
    return report


def analyse_requirements(requirements: list[Requirement]) -> QualityReport:
    """Run quality analysis on a batch of requirements.

    All requirements are sent in a single LLM call so that cross-requirement
    issues (inconsistency, duplication, conflicts) can be detected.
    """
    # Show the agent the WHOLE requirement, not just the statement.
    #
    # Until Day 15 this sent id, priority, stakeholder and statement only. The
    # agent then reliably complained that requirements lacked testable criteria
    # and boundary values -- while `acceptance_criteria` sat in the object,
    # unsent, answering exactly that. One run flagged "how will this be
    # objectively verified?" against a requirement whose acceptance criteria
    # read "renders correctly on screen widths from 320px to 1920px".
    #
    # A reviewer who cannot see half the document will always find it
    # incomplete. That is not a prompt-tuning problem, it is a missing input.
    blocks = []
    for req in requirements:
        lines = [
            f"[{req.id}] priority={req.priority} risk={req.risk_level} "
            f"stakeholder={req.source_stakeholder or 'Not specified'}",
            f"  Statement: {req.statement}",
        ]
        if req.acceptance_criteria:
            lines.append(f"  Acceptance criteria: {req.acceptance_criteria}")
        if req.assumptions:
            lines.append(f"  Assumptions: {req.assumptions}")
        if req.business_justification:
            lines.append(f"  Business justification: {req.business_justification}")
        if req.applicable_regulations:
            lines.append(f"  Applicable regulations: {', '.join(req.applicable_regulations)}")
        if req.dependencies:
            lines.append(f"  Depends on: {', '.join(req.dependencies)}")
        blocks.append("\n".join(lines))

    user_prompt = (
        "Review the following requirements for quality issues.\n"
        "Each requirement is given in full -- read every field before deciding "
        "that something is missing.\n\n"
        + "\n\n".join(blocks)
    )

    report = call_structured(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_model=QualityReport,
        # Greedy decoding. Reviewing requirements is analysis, not generation:
        # the same requirements should yield the same findings, and at the
        # default 0.2 they did not. Four consecutive runs on identical input
        # gave 1/6, 1/6, 1/6 and 0/6 passing, with one run inventing
        # self-contradicting complaints -- quoting "within 2 minutes" and then
        # objecting that the unit of time was unspecified. Sampling variance in
        # a reviewer is indistinguishable from the reviewer being wrong.
        temperature=0.0,
    )
    return filter_issues(report, requirements)
