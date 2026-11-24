"""Use cases (SRS 3.2), the logical data model (3.4) and interface specs (3.1).

The professor's IEEE 830 outline asks for three things a requirement list does
not contain: "detailed specifications of each use case" (3.2), "data entities
and their relationships" (3.4), and "all inputs and outputs" of the external
interfaces (3.1). A model writes them, from the requirements; code decides what
survives:

  * a use case may only claim real functional requirement ids, each once; any
    left over go to an "Other functions" use case, so nothing is lost
  * every actor, external system and entity must be NAMED somewhere in the
    scenario or the requirements -- a model cannot introduce a credit bureau,
    a payment switch or a database table the project never mentioned
  * a relationship must join two entities that survived

Use cases are written in two steps -- first the list, then each one's flows --
because one call for all of them runs past the 2,048-token output cap on a
real scenario. Without an LLM every part falls back to a plain deterministic
version, so an export always completes.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

MAX_USE_CASES = int(os.getenv("SYNTHEIA_MAX_USE_CASES", "6"))

_STOP = {"the", "and", "for", "with", "from", "system", "data", "details", "record",
         "records", "information", "user", "users", "service", "services", "management"}


def corpus_of(scenario: str, requirements: list[dict]) -> str:
    """What the scenario and the requirements SAY -- not who asked for them.

    `source_stakeholder` is deliberately left out: with it in, "Business Owner"
    passed as a user class because "business" was in "Business Analyst", a
    requirement source and not a user (3B, 22 Nov)."""
    parts = [scenario]
    for r in requirements:
        parts += [r.get("statement", ""), r.get("acceptance_criteria", "") or ""]
    return " ".join(parts).lower()


def _distinctive(name: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", (name or "").lower())
            if w not in _STOP and len(w) >= 3]


def grounded(name: str, corpus: str) -> bool:
    """Is every distinctive word of this name in the source?

    ALL, not any: one shared word let "Internet Service Provider" through on
    "internet", and "Business Owner" on "business"."""
    words = _distinctive(name)
    if " ".join(re.findall(r"[a-z0-9]+", (name or "").lower())) in corpus:
        return True
    if not words:
        return False
    for w in words:
        stem = w[:-1] if w.endswith("s") and len(w) > 4 else w
        if not re.search(rf"\b{re.escape(stem)}", corpus):
            return False
    return True


def name_supported(name: str, text: str) -> bool:
    """Does a use-case NAME rest on its own requirements (and the scenario)?

    Looser than `grounded`, because a goal is a verb phrase -- "Verify customer
    identity" rests on "identity verification" -- so words match on a 5-letter
    prefix and a majority is enough. "Submit loan application" for a KYC
    platform matched one word in three and is rejected."""
    words = _distinctive(name)
    if not words:
        return False
    hits = sum(1 for w in words if re.search(rf"\b{re.escape(w[:5] if len(w) >= 6 else w)}", text))
    return hits * 2 > len(words)


# ── LLM-facing schemas ─────────────────────────────────────────────────

class UseCaseHead(BaseModel):
    name: str = Field(description="verb phrase naming the user goal, e.g. 'Submit loan application'")
    primary_actor: str = Field(description="the user class that starts the use case")
    supporting_actors: list[str] = Field(description="external systems or other users involved, as named in the requirements")
    description: str = Field(description="one or two sentences: the goal and its outcome")
    preconditions: list[str] = Field(description="what must be true before it starts")
    postconditions: list[str] = Field(description="what is true after it succeeds")
    requirement_ids: list[str] = Field(description="ids of the functional requirements it realises")


class UseCaseList(BaseModel):
    use_cases: list[UseCaseHead]


class UseCaseFlow(BaseModel):
    main_flow: list[str] = Field(description="3-8 numbered steps of the main success scenario, alternating actor action and system response")
    alternative_flows: list[str] = Field(description="exception or alternative paths, each 'Step N: condition -> what the system does'")


class EntityDraft(BaseModel):
    name: str = Field(description="singular noun, e.g. 'Loan Application'")
    description: str = Field(description="one sentence")
    attributes: list[str] = Field(description="3-8 key attributes")
    identifier: str = Field(description="the attribute that uniquely identifies it")


# Cardinality is asked as two plain questions and the notation is computed:
# asked for "1:N" directly, the 8B wrote "a customer can have multiple KYC
# documents" beside N:1, and "has one risk score" beside 1:N -- it reads the
# notation in either direction.
class RelationshipDraft(BaseModel):
    source: str = Field(description="entity name")
    verb: str = Field(description="1-4 word verb phrase from source to target, e.g. 'uploads', 'is assigned to'")
    target: str = Field(description="entity name")
    targets_per_source: Literal["one", "many"] = Field(
        description="how many target instances ONE source instance can have")
    sources_per_target: Literal["one", "many"] = Field(
        description="how many source instances ONE target instance can belong to")


class DataModelDraft(BaseModel):
    entities: list[EntityDraft]
    relationships: list[RelationshipDraft]


class InterfaceDraft(BaseModel):
    name: str = Field(description="short name, e.g. 'PAN verification'")
    kind: Literal["user", "software", "hardware", "communications"]
    counterpart: str = Field(description="the user class or external system on the other side")
    inputs: str = Field(description="data the system receives through this interface")
    outputs: str = Field(description="data the system sends or displays")
    data_format: str = Field(description="screen, file or message format if the requirements imply one, else 'To be confirmed'")


class InterfaceSet(BaseModel):
    interfaces: list[InterfaceDraft]


# ── What the builder receives ──────────────────────────────────────────

@dataclass
class UseCase:
    id: str
    name: str
    primary_actor: str
    supporting_actors: list[str]
    description: str
    preconditions: list[str]
    main_flow: list[str]
    alternative_flows: list[str]
    postconditions: list[str]
    requirement_ids: list[str]


@dataclass
class DataModel:
    entities: list[tuple[str, str, list[str], str]] = field(default_factory=list)  # name, desc, attrs, id
    relationships: list[tuple[str, str, str, str]] = field(default_factory=list)   # src, rel, tgt, card
    source: str = "none"


@dataclass
class InterfaceSpec:
    rows: list[tuple[str, str, str, str, str, str]] = field(default_factory=list)  # name, kind, party, in, out, fmt
    source: str = "none"


def _reqs_text(reqs: list[dict]) -> str:
    return "\n".join(f"[{r['id']}] {r['statement']}" for r in reqs)


# ── Use cases ──────────────────────────────────────────────────────────

USE_CASE_PROMPT = """\
You are a requirements analyst writing section 3.2 (Functions) of an IEEE 830 \
SRS as use cases. Group the functional requirements below into use cases: one \
per user goal.

Rules:
- Between 2 and {n} use cases. Every requirement id must appear in exactly one \
use case; use only the ids given.
- name: 2-5 words, a verb and an object, using the requirements' own words. \
Name only what the listed requirements describe.
- primary_actor: the person who starts it, chosen from the user classes listed. \
If the system acts on its own, use the user class it acts for.
- supporting_actors: external systems or other users the requirements name. \
Never add a system the requirements do not mention.
- preconditions and postconditions: 1-3 short statements each.
"""

FLOW_PROMPT = """\
Write the main success scenario and the alternative flows for this use case of \
a financial system, using only behaviour the listed requirements state or \
clearly imply.

- main_flow: 3-8 steps, alternating what the actor does and how the system \
responds ("The applicant submits ...", "The system validates ...").
- alternative_flows: 1-4 exception paths, each "Step N: <condition> -> <what \
the system does>". Each condition must follow from THESE requirements; do not \
add generic failures they give no reason to expect.
"""


def _goal(statement: str) -> str:
    """'The system shall validate PAN against NSDL' -> 'Validate PAN against NSDL'."""
    m = re.search(r"\bshall\s+(?:be\s+able\s+to\s+|(?:allow|enable)\s+(?:the\s+|a\s+)?\w+\s+to\s+|"
                  r"let\s+(?:the\s+|a\s+)?\w+\s+)?(.+)",
                  statement, re.I)
    text = (m.group(1) if m else statement).rstrip(". ")
    head = re.split(r"\s+(?:for|based\s+on|against|within|as\s+per|in\s+accordance\s+with|"
                    r"so\s+that|in\s+order\s+to|when|where|which|that|if|before|after|using)\s+",
                    text, maxsplit=1, flags=re.I)[0]
    words = (head if len(head.split()) >= 3 else text).split()[:9]
    while words and words[-1].lower() in {"a", "an", "the", "of", "for", "to", "and", "or", "in", "with", "by"}:
        words.pop()
    text = " ".join(words)
    return text[:1].upper() + text[1:] if text else "Function"


def fallback_use_cases(functional: list[dict]) -> list[UseCase]:
    """One use case per functional requirement, named from its verb phrase.

    Offline, there is nothing to group by meaning; grouping by stakeholder made
    use cases named after whoever ASKED for a requirement, which is not a user
    goal. One requirement per use case is plainer but true."""
    from src.generators.srs.narrative import named_user_classes

    out = []
    for i, r in enumerate(functional[:MAX_USE_CASES], start=1):
        users = named_user_classes(r["statement"], [])
        out.append(UseCase(
            id=f"UC-{i:02d}", name=_goal(r["statement"]),
            primary_actor=users[0][0] if users else "To be confirmed",
            supporting_actors=[], description=r["statement"].rstrip(".") + ".",
            preconditions=[], main_flow=[], alternative_flows=[], postconditions=[],
            requirement_ids=[r["id"]]))
    rest = functional[MAX_USE_CASES:]
    if rest:
        out.append(UseCase(
            id=f"UC-{len(out) + 1:02d}", name="Other functions", primary_actor="User",
            supporting_actors=[], description="Functional requirements not assigned to a use case above.",
            preconditions=[], main_flow=[], alternative_flows=[], postconditions=[],
            requirement_ids=[r["id"] for r in rest]))
    return out


def _actor(proposed: str, corpus: str, statements: list[str]) -> str:
    """The model's actor if the text names it and it is a person; otherwise the
    first person the use case's own requirements name; otherwise unknown."""
    from src.generators.srs.narrative import named_user_classes

    proposed = (proposed or "").strip()
    people = {u[0].lower() for u in named_user_classes(corpus, [])}
    if proposed and grounded(proposed, corpus) and \
            any(p in proposed.lower() or proposed.lower().rstrip("s") in p for p in people):
        return proposed
    named = named_user_classes(" ".join(statements), [])
    return named[0][0] if named else "To be confirmed"


def write_use_cases(functional: list[dict], scenario: str, all_reqs: list[dict],
                    use_llm: bool = True) -> tuple[list[UseCase], str]:
    """Use cases covering every functional requirement exactly once."""
    if not functional:
        return [], "none"
    if not use_llm:
        return fallback_use_cases(functional), "fallback"
    from src.utils.llm import call_structured

    from src.generators.srs.narrative import named_user_classes

    corpus = corpus_of(scenario, all_reqs)
    users = [u[0] for u in named_user_classes(scenario, all_reqs)]
    try:
        head: UseCaseList = call_structured(
            system_prompt=USE_CASE_PROMPT.format(n=MAX_USE_CASES),
            user_prompt=(f"Problem scenario:\n{scenario}\n\nUser classes: "
                         f"{', '.join(users) or 'not named -- infer from the scenario'}"
                         f"\n\nFunctional requirements:\n{_reqs_text(functional)}"),
            response_model=UseCaseList, temperature=0.0)
    except Exception as e:                                   # noqa: BLE001
        print(f"  (SRS use cases: LLM unavailable -- {type(e).__name__}; grouping by stakeholder)")
        return fallback_use_cases(functional), "fallback"

    valid = {r["id"]: r for r in functional}
    taken: set[str] = set()
    cases: list[UseCase] = []
    for h in head.use_cases[:MAX_USE_CASES]:
        ids = [i for i in h.requirement_ids if i in valid and i not in taken]
        if not ids or not h.name.strip():
            continue
        taken.update(ids)
        own = " ".join([scenario] + [valid[i]["statement"] for i in ids]).lower()
        name, desc = h.name.strip(), h.description.strip()
        if not name_supported(name, own):        # a goal the requirements never describe
            name, desc = _goal(valid[ids[0]]["statement"]), ""
        actor = _actor(h.primary_actor, corpus, [valid[i]["statement"] for i in ids])
        cases.append(UseCase(
            id="", name=name, primary_actor=actor,
            supporting_actors=[a.strip() for a in h.supporting_actors
                               if a.strip() and a.strip() != actor and grounded(a, corpus)][:4],
            description=desc or "; ".join(valid[i]["statement"].rstrip(".") for i in ids[:2]) + ".",
            preconditions=[p for p in h.preconditions if p.strip()][:3],
            main_flow=[], alternative_flows=[], postconditions=[p for p in h.postconditions if p.strip()][:3],
            requirement_ids=ids))
    # Requirements the model left out each become a use case of their own,
    # named from their verb phrase -- a lump called "Other functions" with
    # actor "User" is what a weak model produces and says nothing. Only past a
    # cap do the rest share one.
    rest = [i for i in valid if i not in taken]
    room = max(0, MAX_USE_CASES + 3 - len(cases))
    for i in rest[:room]:
        st = valid[i]["statement"]
        cases.append(UseCase(id="", name=_goal(st), primary_actor=_actor("", corpus, [st]),
                             supporting_actors=[], description=st.rstrip(".") + ".",
                             preconditions=[], main_flow=[], alternative_flows=[], postconditions=[],
                             requirement_ids=[i]))
    if rest[room:]:
        cases.append(UseCase(id="", name="Other functions", primary_actor="To be confirmed",
                             supporting_actors=[],
                             description="Functional requirements not assigned to a use case above.",
                             preconditions=[], main_flow=[],
                             alternative_flows=[], postconditions=[], requirement_ids=rest[room:]))

    for i, uc in enumerate(cases, start=1):
        uc.id = f"UC-{i:02d}"
        if uc.name == "Other functions":                     # leftovers: no single goal to narrate
            continue
        reqs = [valid[r] for r in uc.requirement_ids]
        try:
            flow: UseCaseFlow = call_structured(
                system_prompt=FLOW_PROMPT,
                user_prompt=(f"Use case: {uc.name}\nPrimary actor: {uc.primary_actor}\n"
                             f"Supporting actors: {', '.join(uc.supporting_actors) or 'none'}\n"
                             f"Description: {uc.description}\n\nRequirements:\n{_reqs_text(reqs)}"),
                response_model=UseCaseFlow, temperature=0.0)
            # The renderer numbers the steps; the model often numbers them too
            # ("1. Step 1: The customer ...").
            steps = (re.sub(r"^\s*(?:step\s*\d+\s*[.):-]?|\d+\s*[.):-])\s*(?:step\s*\d+\s*[.):-]\s*)?",
                            "", s, flags=re.I).strip() for s in flow.main_flow)
            uc.main_flow = [s for s in steps if s][:8]
            uc.alternative_flows = [s.strip() for s in flow.alternative_flows if s.strip()][:4]
        except Exception:                                    # noqa: BLE001
            pass                                             # rendered as "see the requirements"
    return cases, "llm"


# ── Logical data model ─────────────────────────────────────────────────

DATA_MODEL_PROMPT = """\
You are writing section 3.4 (Logical database requirements) of an IEEE 830 SRS: \
the data entities the system stores and their relationships, derived from the \
requirements.

Rules:
- 3 to 8 entities, each a thing the requirements say the system stores or \
processes (customer, application, document, decision, log...). Use names the \
requirements use. Do not invent entities they never mention.
- attributes: 3-8 key fields. identifier: the field that identifies an instance.
- relationships: how the entities relate. Write each from the owning side: \
source is the entity that has, creates or receives the other ('Customer has \
KYC Document', 'Customer receives Notification'), never 'KYC Document belongs \
to Customer'. For each, say how many targets one source can have, and how many \
sources one target can belong to.
"""

# (targets per source, sources per target) -> notation, read source:target.
_CARD = {("one", "one"): "1:1", ("many", "one"): "1:N",
         ("one", "many"): "N:1", ("many", "many"): "M:N"}
_GLOSS = {"1:1": "one {s} to one {t}", "1:N": "one {s} to many {t}",
          "N:1": "many {s} to one {t}", "M:N": "many {s} to many {t}"}


def write_data_model(scenario: str, reqs: list[dict], use_llm: bool = True) -> DataModel:
    if not use_llm or not reqs:
        return DataModel()
    from src.utils.llm import call_structured

    corpus = corpus_of(scenario, reqs)
    try:
        out: DataModelDraft = call_structured(
            system_prompt=DATA_MODEL_PROMPT,
            user_prompt=f"Problem scenario:\n{scenario}\n\nRequirements:\n{_reqs_text(reqs)}",
            response_model=DataModelDraft, temperature=0.0)
    except Exception as e:                                   # noqa: BLE001
        print(f"  (SRS data model: LLM unavailable -- {type(e).__name__})")
        return DataModel()
    ents, names = [], set()
    for e in out.entities[:8]:
        n = e.name.strip()
        if n and n.lower() not in names and grounded(n, corpus):
            names.add(n.lower())
            ents.append((n, e.description.strip(), [a.strip() for a in e.attributes if a.strip()][:8],
                         e.identifier.strip()))
    shape = {n.lower(): ({a.lower() for a in attrs}, ident.lower()) for n, _, attrs, ident in ents}

    def holds_key(holder: str, other: str) -> bool:
        # A column holding another entity's identifier holds ONE of them. A
        # generic "id" shared by both is the holder's own key, not a reference.
        attrs, own = shape[holder.lower()]
        key = shape[other.lower()][1]
        return bool(key) and key != own and key in attrs

    rels = []
    for r in out.relationships:
        s, t = r.source.strip(), r.target.strip()
        if s.lower() in names and t.lower() in names and s.lower() != t.lower():
            per_s, per_t = r.targets_per_source, r.sources_per_target
            if holds_key(t, s):
                per_t = "one"
            if holds_key(s, t):
                per_s = "one"
            card = _CARD[(per_s, per_t)]
            rels.append((s, r.verb.strip(), t, f"{card} ({_GLOSS[card].format(s=s, t=t)})"))
    return DataModel(entities=ents, relationships=rels, source="llm" if ents else "none")


# ── External interfaces ────────────────────────────────────────────────

INTERFACE_PROMPT = """\
You are writing section 3.1 (External interfaces) of an IEEE 830 SRS: every \
point where the system exchanges data with a user or another system, with its \
inputs and outputs.

Rules:
- One entry per interface the requirements imply: user screens (kind "user"), \
external systems and APIs ("software"), devices ("hardware"), messaging such as \
SMS or e-mail ("communications").
- counterpart: the user class or external system, as named in the requirements.
- inputs / outputs: the data exchanged. data_format: the screen, file or \
message format if the requirements imply one; otherwise "To be confirmed".
- Do not invent systems the requirements do not mention.
"""


def write_interfaces(scenario: str, reqs: list[dict], use_llm: bool = True) -> InterfaceSpec:
    if not use_llm or not reqs:
        return InterfaceSpec()
    from src.utils.llm import call_structured

    corpus = corpus_of(scenario, reqs)
    try:
        out: InterfaceSet = call_structured(
            system_prompt=INTERFACE_PROMPT,
            user_prompt=f"Problem scenario:\n{scenario}\n\nRequirements:\n{_reqs_text(reqs)}",
            response_model=InterfaceSet, temperature=0.0)
    except Exception as e:                                   # noqa: BLE001
        print(f"  (SRS interfaces: LLM unavailable -- {type(e).__name__})")
        return InterfaceSpec()
    from src.generators.srs.narrative import named_user_classes

    people = {u[0].lower() for u in named_user_classes(corpus, [])}
    rows = []
    for i in out.interfaces[:12]:
        party = i.counterpart.strip()
        if not i.name.strip() or not grounded(party, corpus):
            continue
        kind = "user" if party.lower().rstrip("s") in people else i.kind
        rows.append((i.name.strip(), kind, party, i.inputs.strip(), i.outputs.strip(),
                     i.data_format.strip() or "To be confirmed"))
    return InterfaceSpec(rows=rows, source="llm" if rows else "none")
