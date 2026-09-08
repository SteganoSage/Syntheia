"""Quality analysis agent — checks requirements for ambiguity, incompleteness,
inconsistency, and other quality issues.

Day 6: LLM-based quality checker. Sends ALL requirements in one call so
cross-requirement issues (duplication, inconsistency) can be detected.
"""

from src.models.quality import QualityReport
from src.models.requirement import Requirement
from src.utils.llm import call_structured

SYSTEM_PROMPT = """\
You are a senior QA engineer reviewing software requirements for a financial \
system. Analyse EACH requirement for the following quality issues:

1. **Ambiguity** — Vague or unmeasurable terms like "fast", "quickly", \
"user-friendly", "efficient", "appropriate", "etc.". Every requirement must \
have a concrete, measurable definition.
2. **Incompleteness** — Missing critical details: who performs the action, \
what triggers it, what the expected outcome is, or missing boundary values.
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
8. **Undefined Terminology** — Uses domain jargon, acronyms, or technical \
terms without defining them (e.g. "eKYC", "T+1", "PEP" should be defined \
or referenced).
9. **Conflicting Expectations** — Two different stakeholders want opposite \
behaviours or contradictory priorities.
10. **Missing security/compliance** — The requirement handles sensitive data \
(PII, financial data, biometrics) but does not mention encryption, access \
control, or applicable regulations.

Rules:
- Check EVERY requirement in the list — do not skip any.
- If a requirement passes ALL 10 checks, set passed=True and return an \
empty issues list for that requirement.
- If a requirement has problems, set passed=False and list each issue with \
a specific, actionable suggestion for how to fix it.
- For cross-requirement issues (Inconsistency, Duplication, Conflicting \
Expectations), reference the conflicting requirement IDs in the description.
- Be strict but fair — flag real problems, not nitpicks.
"""


def analyse_requirements(requirements: list[Requirement]) -> QualityReport:
    """Run quality analysis on a batch of requirements.

    All requirements are sent in a single LLM call so that cross-requirement
    issues (inconsistency, duplication, conflicts) can be detected.
    """
    # Build a formatted list of all requirements for the LLM
    req_lines = []
    for req in requirements:
        req_lines.append(
            f"[{req.id}] (Priority: {req.priority}, "
            f"Stakeholder: {req.source_stakeholder or 'Not specified'}) "
            f"{req.statement}"
        )

    user_prompt = (
        "Review the following requirements for quality issues:\n\n"
        + "\n".join(req_lines)
    )

    return call_structured(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_model=QualityReport,
    )
