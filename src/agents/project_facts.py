"""Catch "requirements" that are really facts about the project, not the system.

The extraction prompt already says it, in so many words (problem log 6.1):
team size, staffing, delivery timeline, budget, funding and methodology are
the PROJECT, not the software. On 14 Nov the 8B ignored it anyway and produced
"The system shall be developed within 18 months" and "The system shall not
require changes to requirements once signed off" from the insurance scenario.
A rule the model can be told and still break is a rule to enforce in code.

Narrow on purpose. Dropping a real requirement is worse than keeping a stray
project fact -- the second is visible and a reviewer deletes it, the first is
silently missing. So every pattern needs the project's own vocabulary, not
just a word that also appears in system behaviour:

    kept   "allow the compliance team to review flagged cases"   (users, not staff)
    kept   "process loan applications within 30 days"           (system behaviour)
    kept   "let users set a monthly budget"                     (a budgeting feature)
    kept   "require sign-off from a senior officer above 10 lakh"
    gone   "be developed within 18 months"
    gone   "be built by a team of 3 developers"
    gone   "be developed using agile methodology"

Nothing is lost by dropping them: the SDLC engine reads timeline, team and
funding from the scenario itself (`infer_project_context`), not from here.
"""

from __future__ import annotations

import re

_N = r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|twelve|fifteen|eighteen|twenty)"
_BUILD = r"(?:developed|built|completed|delivered|implemented|launched|released|deployed|ready|live|finished)"

PATTERNS: list[tuple[str, re.Pattern]] = [
    ("delivery timeline", re.compile(
        rf"\b(?:be|go)\s+{_BUILD}\s+(?:with)?in\s+(?:the\s+next\s+)?{_N}\s+(?:days?|weeks?|months?|years?|quarters?)\b"
        rf"|\b(?:be|go)\s+{_BUILD}\s+by\s+(?:the\s+end\s+of\s+)?(?:q[1-4]|january|february|march|april|may|june|july|"
        rf"august|september|october|november|december|diwali|\d{{4}})\b"
        rf"|\bgo[\s-]live\s+(?:with)?in\s+{_N}\s+(?:weeks?|months?)\b", re.I)),
    ("team and staffing", re.compile(
        rf"\b{_N}\s+(?:developers?|engineers?|programmers?|testers?)\b"
        r"|\b(?:development|project|delivery|engineering)\s+team\b"
        r"|\bteam\s+of\s+" + _N + r"\b", re.I)),
    ("budget and funding", re.compile(
        r"\b(?:project|development|delivery)\s+(?:budget|cost|funding)\b"
        r"|\bfunding\s+(?:is|was|has\s+been|will\s+be)\s+(?:approved|allocated|available)\b"
        r"|\bwithin\s+(?:a|the)\s+budget\s+of\b", re.I)),
    ("development methodology", re.compile(
        r"\b(?:developed|built|delivered|implemented|managed)\s+(?:using|with|following|under)\s+"
        r"(?:an?\s+|the\s+)?(?:agile|scrum|kanban|waterfall|v[\s-]?model|v[\s-]?shaped|spiral|"
        r"iterative|incremental|devops|devsecops)\b"
        r"|\b(?:agile|scrum|waterfall|spiral)\s+(?:methodolog\w*|process|approach)\b", re.I)),
    ("requirements process", re.compile(
        r"\bchanges?\s+to\s+(?:the\s+|any\s+)?(?:existing\s+)?requirements\b"
        r"|\brequirements?\s+(?:shall|will|must|do|does)?\s*(?:not\s+)?(?:change|be\s+changed)\b"
        r"|\brequirements?\s+(?:are|is|have\s+been|will\s+be|once)\s+(?:\w+\s+)?signed[\s-]off\b", re.I)),
]


def project_fact(statement: str) -> str:
    """The kind of project fact this statement is, or '' if it is a real requirement."""
    for reason, pattern in PATTERNS:
        if pattern.search(statement or ""):
            return reason
    return ""
