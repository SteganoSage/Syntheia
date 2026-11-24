"""The parts of an SRS a model writes better than code -- and nothing else.

Two calls, both gated:

  NARRATIVE  product name, purpose, scope, product perspective, and the
             objectives and constraints the scenario states. Every objective
             and constraint must quote the scenario; one that cannot is
             dropped, the same evidence standard the SDLC engine and Phase B
             hold the model to.
  FEATURES   functional requirements grouped into IEEE 830 "system features"
             (§A.5: organised by feature) with stimulus/response sequences.
             Only real requirement ids are accepted, each in one feature; any
             left out land in "Other functions" so nothing is lost.

Everything else in the SRS -- sections, references, glossary, traceability,
verification -- is assembled in code from the run. Without an LLM both calls
fall back to deterministic text, so a saved run can always be exported.
"""

from __future__ import annotations

import re

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from src.utils.evidence import quotes_source


# ── LLM-facing schemas (no defaults: a defaulted field is optional in the
#    generated grammar and gets omitted -- problem log 3.2) ──────────────

class QuotedItem(BaseModel):
    text: str = Field(description="the objective or constraint, one sentence")
    quote: str = Field(description="the exact words from the problem scenario it rests on")


class UserClassDraft(BaseModel):
    name: str = Field(description="the user class, as named in the scenario or requirements")
    description: str = Field(description="who they are and what they do with the system (their skill level goes in technical_expertise, not here)")
    technical_expertise: str = Field(description="Low, Medium or High, with a short reason")
    frequency_of_use: str = Field(description="e.g. daily, occasional, once per application")


class SRSNarrative(BaseModel):
    product_name: str = Field(description="2-6 words naming the system, from the scenario's own words")
    purpose: str = Field(description="2-3 sentences: which system this SRS specifies, for whom, and who should read it")
    scope: str = Field(description="3-5 sentences: what the system will do and the benefits the scenario states")
    product_perspective: str = Field(description="2-3 sentences: the business case; new or replacing/extending an existing system; external systems it works with, as named in the scenario")
    operational_concept: str = Field(description="2-4 sentences: how the system is used day to day -- who does what, in what order")
    objectives: list[QuotedItem] = Field(description="business objectives the scenario states")
    constraints: list[QuotedItem] = Field(description="limits the scenario places on the solution")
    out_of_scope: list[QuotedItem] = Field(description="what the system will NOT do; quote the scenario if it says so, otherwise an empty quote")
    user_classes: list[UserClassDraft] = Field(description="each class of user named in the scenario or requirements")


class FeatureDraft(BaseModel):
    name: str = Field(description="2-5 words")
    description: str = Field(description="1-2 sentences: what the feature does and for whom")
    requirement_ids: list[str] = Field(description="ids of the requirements in this feature")
    stimulus_response: list[str] = Field(description="2-4 lines of the form 'actor action -> system response'")


class FeatureSet(BaseModel):
    features: list[FeatureDraft]


# ── What the builder receives ──────────────────────────────────────────

@dataclass
class Narrative:
    product_name: str
    purpose: str
    scope: str
    perspective: str
    objectives: list[tuple[str, str]] = field(default_factory=list)   # (text, quote)
    constraints: list[tuple[str, str]] = field(default_factory=list)
    operational_concept: str = ""
    # (text, quote): a quote means the scenario says so; "" means assumed.
    out_of_scope: list[tuple[str, str]] = field(default_factory=list)
    # (name, description, technical expertise, frequency of use)
    user_classes: list[tuple[str, str, str, str]] = field(default_factory=list)
    source: str = "fallback"            # "llm" | "fallback"
    dropped: int = 0                    # items rejected for an untraceable quote


@dataclass
class Feature:
    name: str
    description: str
    requirement_ids: list[str]
    stimulus_response: list[str] = field(default_factory=list)


NARRATIVE_PROMPT = """\
You write the narrative sections of a Software Requirements Specification \
(IEEE Std 830) for a financial-services system. You are given the problem \
scenario and the requirements already extracted from it.

Write ONLY what the scenario and the requirements support. Do not invent \
organisations, products, technologies, numbers, dates or external systems that \
are not mentioned. Where something is not stated, say it is to be confirmed.

- product_name: 2-6 words naming the system in the scenario's own words, for \
example "Digital Customer Onboarding Platform". No marketing names.
- purpose: which system this SRS specifies, for which organisation (as the \
scenario describes it), and that it is written for business owners, \
developers, testers and compliance reviewers.
- scope: what the system will do and the benefits the scenario gives. State \
what is out of scope only if the scenario says so.
- product_perspective: whether the system is new, or replaces or extends an \
existing one, and which external systems it works with -- only those named.
- objectives: the business objectives the scenario states, each with the \
exact words it rests on in `quote`. Empty if none are stated.
- constraints: limits the scenario places on the solution -- regulation, \
technology, schedule, staffing, budget -- each with the exact words in `quote`.
- operational_concept: how the system is used day to day: who does what, in \
what order, from the requirements.
- out_of_scope: what the system will NOT do. If the scenario says so, quote it. \
Otherwise list only exclusions a reader would reasonably assume from the scope \
(they will be marked as assumptions) and leave `quote` empty.
- user_classes: every class of user named in the scenario or the requirements' \
stakeholders, with their technical expertise (Low / Medium / High and why) and \
how often they use the system. Do not invent user classes.
"""

FEATURES_PROMPT = """\
Group these functional requirements into the system features of section 3.2 \
of an IEEE 830 SRS. A feature is a coherent service the system provides, such \
as "Customer onboarding" or "Loan decisioning".

Rules:
- Every requirement id given must appear in exactly one feature. Use only the \
ids given.
- Between 2 and 6 features. A feature may hold a single requirement.
- name: 2-5 words. description: 1-2 sentences on what it does and for whom.
- stimulus_response: 2-4 lines "actor action -> system response" describing \
how the feature is used, using only behaviour the requirements state.
"""


# ── Narrative ──────────────────────────────────────────────────────────

# People who USE a financial system, as scenarios name them. A requirement's
# source_stakeholder is who asked for it (a business analyst, a risk team),
# which is not the same thing, so 2.3 is not built from that field.
_USER_TERMS = ["customer", "applicant", "borrower", "policyholder", "claimant", "cardholder",
               "merchant", "agent", "underwriter", "loan officer", "credit officer",
               "compliance officer", "relationship manager", "branch staff", "operations staff",
               "operations team", "administrator", "auditor", "adjuster", "teller", "investor",
               "approver", "reviewer", "call centre agent", "support staff"]


def named_user_classes(scenario: str, requirements: list[dict]) -> list[tuple[str, str, str, str]]:
    text = " ".join([scenario] + [r.get("statement", "") for r in requirements])
    found = []
    for term in sorted(_USER_TERMS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(term)}s?\b", text, re.I) and \
                not any(term in f for f in found):
            found.append(term)
    where = lambda t: "the scenario" if re.search(rf"\b{re.escape(t)}", scenario, re.I) else "the requirements"
    return [(t.title(), f"Named in {where(t)}.", "To be confirmed", "To be confirmed")
            for t in sorted(found)]


def fallback_narrative(scenario: str, requirements: list[dict]) -> Narrative:
    from src.generators.srs.derive import fallback_product_name, scenario_sentences

    name = fallback_product_name(scenario)
    first = " ".join(scenario_sentences(scenario)[:2])
    return Narrative(
        product_name=name,
        purpose=(f"This Software Requirements Specification defines the requirements "
                 f"of the {name}. It is written for the business owners, developers, "
                 f"testers and compliance reviewers of the system."),
        scope=f"The source scenario describes the system as follows: “{first}”",
        operational_concept=("How the system is used day to day is not described in the "
                             "source scenario and is to be confirmed; the use cases in "
                             "section 3.2 describe each user goal."),
        user_classes=named_user_classes(scenario, requirements),
        perspective=("Whether the system is new or extends an existing platform is not "
                     "stated in the source scenario and is to be confirmed. The external "
                     "systems it depends on, as named in the requirements, are listed in "
                     "the subsections below."),
    )


def _requirements_text(requirements: list[dict]) -> str:
    return "\n".join(f"[{r['id']}] {r['statement']}" for r in requirements)


def write_narrative(scenario: str, requirements: list[dict]) -> Narrative:
    """LLM narrative, gated; falls back to deterministic text on any failure."""
    fb = fallback_narrative(scenario, requirements)
    try:
        from src.utils.llm import call_structured

        out: SRSNarrative = call_structured(
            system_prompt=NARRATIVE_PROMPT,
            user_prompt=(f"Problem scenario:\n{scenario}\n\n"
                         f"Requirements:\n{_requirements_text(requirements)}"),
            response_model=SRSNarrative, temperature=0.0)
    except Exception as e:                                   # noqa: BLE001
        print(f"  (SRS narrative: LLM unavailable -- {type(e).__name__}; "
              f"using deterministic text)")
        return fb

    def gate(items):
        kept = [(i.text.strip(), i.quote.strip()) for i in items
                if i.text.strip() and quotes_source(i.quote, scenario)]
        return kept, len(items) - len(kept)

    objectives, d1 = gate(out.objectives)
    constraints, d2 = gate(out.constraints)
    name = out.product_name.strip().strip('"“”\'')
    if not name or len(name.split()) > 8:
        name = fb.product_name
    def prose(text: str, minimum: int, fallback: str) -> str:
        # A small model sometimes answers a prose field with one word -- the
        # 3B returned "new" for product_perspective (22 Nov). A section of an
        # SRS that is one word long is worse than the plain fallback.
        text = " ".join((text or "").split())
        return text if len(text.split()) >= minimum else fallback

    from src.generators.srs.analysis_models import corpus_of, grounded

    corpus = corpus_of(scenario, requirements)
    # Out of scope: a quote that is really in the scenario makes it a stated
    # exclusion; an invented quote is dropped; an empty one is an assumption.
    out_scope, d3 = [], 0
    for i in out.out_of_scope[:6]:
        # The list is introduced by "The system will not:", so the model's own
        # "Assumption:" / "The system will not" prefixes repeat it, and a
        # placeholder ("To be confirmed") is not an exclusion at all.
        text = re.sub(r"^(?:assum(?:ption|ed)\s*:\s*)?(?:the\s+system\s+(?:will|shall)\s+not\s+)?",
                      "", i.text.strip(), flags=re.I).strip().rstrip(".")
        text = text[:1].upper() + text[1:]
        quote = i.quote.strip()
        if not text or re.fullmatch(r"(?:to be confirmed|tbd|none|n/?a|not (?:stated|specified))",
                                    text, re.I):
            continue
        if quote and not quotes_source(quote, scenario):
            d3 += 1
            continue
        out_scope.append((text, quote))
    # A 3B writes the skill level into the description ("Low"); a description
    # under four words describes nothing, so it falls back to where the class
    # was found. The 8B writes the expertise REASON there instead ("Low, as
    # they only upload documents") -- that belongs in the expertise column.
    found = {u[0].lower(): u[1] for u in named_user_classes(scenario, requirements)}
    level = re.compile(r"(?:low|medium|moderate|high|basic|intermediate|advanced)\b", re.I)
    classes = []
    for u in out.user_classes:
        if not (u.name.strip() and grounded(u.name, corpus)):
            continue
        desc, skill = u.description.strip(), u.technical_expertise.strip()
        m = level.match(desc)
        if m:
            # "Low, as they will only need to upload documents" -> expertise
            # "Low", description "Upload documents": the reason IS what they do.
            does = re.sub(r"^[\s,;:.\-–—]*(?:as|because|since)?\s*(?:they|users?)?\s*(?:will|would)?\s*"
                          r"(?:only\s+)?(?:need\s+to|have\s+to)?\s*", "", desc[m.end():], flags=re.I)
            does = does.strip().rstrip(".")
            if len(does.split()) >= 3:
                skill = skill if skill and len(skill.split()) == 1 else m.group(0).capitalize()
                desc = does[:1].upper() + does[1:] + "."
            else:
                skill = desc if len(skill.split()) <= 1 else skill
                desc = ""
        elif len(desc.split()) < 4:
            desc = ""
        if not desc:
            desc = found.get(u.name.strip().lower(), "Named in the source material.")
        classes.append((u.name.strip(), desc, skill or "To be confirmed",
                        u.frequency_of_use.strip() or "To be confirmed"))
    return Narrative(
        product_name=name,
        purpose=prose(out.purpose, 12, fb.purpose),
        scope=prose(out.scope, 15, fb.scope),
        perspective=prose(out.product_perspective, 10, fb.perspective),
        operational_concept=prose(out.operational_concept, 12, fb.operational_concept),
        objectives=objectives, constraints=constraints,
        out_of_scope=out_scope, user_classes=classes or fb.user_classes,
        source="llm", dropped=d1 + d2 + d3,
    )


# ── Features ───────────────────────────────────────────────────────────

def fallback_features(functional: list[dict]) -> list[Feature]:
    """One feature per originating stakeholder -- crude, but honest and complete."""
    groups: dict[str, list[str]] = {}
    for r in functional:
        who = (r.get("source_stakeholder") or "General").strip() or "General"
        groups.setdefault(who, []).append(r["id"])
    return [Feature(name=f"Functions for {who}",
                    description=f"Functional requirements originating with the {who}.",
                    requirement_ids=ids)
            for who, ids in groups.items()]


def group_features(functional: list[dict], use_llm: bool = True) -> tuple[list[Feature], str]:
    """Features covering every functional requirement exactly once."""
    if not functional:
        return [], "none"
    if not use_llm:
        return fallback_features(functional), "fallback"
    try:
        from src.utils.llm import call_structured

        out: FeatureSet = call_structured(
            system_prompt=FEATURES_PROMPT,
            user_prompt=f"Functional requirements:\n{_requirements_text(functional)}",
            response_model=FeatureSet, temperature=0.0)
    except Exception as e:                                   # noqa: BLE001
        print(f"  (SRS features: LLM unavailable -- {type(e).__name__}; "
              f"grouping by stakeholder)")
        return fallback_features(functional), "fallback"

    valid = [r["id"] for r in functional]
    taken: set[str] = set()
    features = []
    for f in out.features:
        ids = [i for i in f.requirement_ids if i in valid and i not in taken]
        taken.update(ids)
        if ids and f.name.strip():
            features.append(Feature(name=f.name.strip(), description=f.description.strip(),
                                    requirement_ids=ids,
                                    stimulus_response=[s.strip() for s in f.stimulus_response
                                                       if s.strip()][:5]))
    rest = [i for i in valid if i not in taken]
    if rest:
        features.append(Feature(name="Other functions",
                                description="Functional requirements not assigned to a "
                                            "feature above.",
                                requirement_ids=rest))
    return features, "llm"
