"""Load the SDLC selection framework and score models against it.

No LLM here, and that is the point. The supervisor's tables already contain the
scoring function: each cell is the answer a model expects, so scoring is
counting matches. Asking a language model to do that arithmetic would make the
one auditable step in the pipeline non-deterministic, and the Definition of Done
requires the justification to reference actual selection criteria -- which means
being able to show the working.

The split across the whole agent is:

    LLM     infers the answers from requirements   (language judgement)
    HERE    counts matches and ranks               (arithmetic)
    LLM     writes the justification               (explanation)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
FRAMEWORK_PATH = REPO / "data/sdlc/framework.yaml"


class FrameworkError(ValueError):
    """The framework file is unusable. Always fatal -- never scored around."""


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    table: str
    table_name: str
    source: str                      # "inferred" | "supplied"
    expects: dict[str, bool]
    agile_rationale: str = ""
    # How to read the question. Shipped to the model with the question text --
    # the supervisor's wording is terse, and several questions were being
    # answered as though they asked something else.
    gloss: str = ""
    # "either" | "scenario". Which source can legitimately support an answer.
    # Requirements cannot testify about budgets or product lifespan; both models
    # cited requirement ids for exactly those, and the citations passed every
    # structural check because the ids existed.
    evidence_from: str = "either"
    # "supervisor_framework" | "derived" -- same distinction the model columns
    # carry. A question we added is a change to the rubric being scored against,
    # so the output has to be able to say which questions were his and which
    # were ours rather than presenting all of them as transcription.
    origin: str = "supervisor_framework"

    @property
    def is_derived(self) -> bool:
        return self.origin != "supervisor_framework"


@dataclass(frozen=True)
class Model:
    id: str
    name: str
    source: str                      # "supervisor_framework" | "derived"
    basis: str = ""

    @property
    def is_derived(self) -> bool:
        return self.source != "supervisor_framework"


@dataclass
class ModelScore:
    model: Model
    matched: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)

    @property
    def answered(self) -> int:
        return len(self.matched) + len(self.missed)

    @property
    def score(self) -> float:
        """Fraction of answered questions this model's column agrees with.

        A fraction rather than a count, because tables can be skipped when their
        input is not available. Raw counts would let a model scored on 10
        questions look worse than one scored on 19 for no reason but coverage.
        """
        return len(self.matched) / self.answered if self.answered else 0.0


class Framework:
    def __init__(self, models: list[Model], questions: list[Question], version: str):
        self.models = models
        self.questions = questions
        self.version = version
        self._by_id = {m.id: m for m in models}

    # ── Loading ────────────────────────────────────────────────────────

    @classmethod
    def load(cls, path: Path | None = None) -> "Framework":
        data = yaml.safe_load(Path(path or FRAMEWORK_PATH).read_text(encoding="utf-8"))

        models = [Model(id=m["id"], name=m["name"], source=m["source"],
                        basis=m.get("basis", "")) for m in data["models"]]
        model_ids = {m.id for m in models}
        if len(model_ids) != len(models):
            raise FrameworkError("duplicate model ids in the framework")

        questions: list[Question] = []
        for table in data["tables"]:
            for q in table["questions"]:
                expects = {k: bool(v) for k, v in q["expects"].items()}
                # Strict on purpose. A model missing an answer would be scored
                # on a smaller denominator than its rivals and could win on
                # coverage rather than fit -- silently, since every individual
                # number would still look reasonable.
                missing = model_ids - set(expects)
                extra = set(expects) - model_ids
                if missing or extra:
                    raise FrameworkError(
                        f"question {q['id']}: "
                        + (f"no answer for {sorted(missing)}. " if missing else "")
                        + (f"unknown model(s) {sorted(extra)}. " if extra else "")
                        + "Every model must have an answer to every question; "
                          "add the missing keys to data/sdlc/framework.yaml."
                    )
                questions.append(Question(
                    id=q["id"], text=q["text"], table=table["id"],
                    table_name=table["name"], source=table["source"],
                    expects=expects, agile_rationale=q.get("agile_rationale", ""),
                    origin=q.get("origin", "supervisor_framework"),
                    gloss=q.get("gloss", "").strip(),
                    evidence_from=q.get("evidence_from", "either"),
                ))

        if not questions:
            raise FrameworkError("framework contains no questions")
        return cls(models, questions, str(data.get("version", "unknown")))

    # ── Access ─────────────────────────────────────────────────────────

    def questions_for(self, source: str) -> list[Question]:
        return [q for q in self.questions if q.source == source]

    def tables(self) -> dict[str, list[Question]]:
        out: dict[str, list[Question]] = {}
        for q in self.questions:
            out.setdefault(q.table, []).append(q)
        return out

    def question(self, qid: str) -> Question | None:
        return next((q for q in self.questions if q.id == qid), None)

    # ── Scoring ────────────────────────────────────────────────────────

    def score(self, answers: dict[str, bool]) -> list[ModelScore]:
        """Rank every model against the project's answers, best first.

        `answers` maps question id -> the project's yes/no. Unknown ids are
        ignored; unanswered questions simply do not count for anyone, which is
        how a skipped table stays neutral rather than penalising models whose
        column happened to say "no".
        """
        scores = [ModelScore(model=m) for m in self.models]
        for q in self.questions:
            if q.id not in answers:
                continue
            given = bool(answers[q.id])
            for s in scores:
                (s.matched if q.expects[s.model.id] == given else s.missed).append(q.id)

        # Ties broken by name so a rerun cannot reorder equal scores -- the
        # ranking has to be reproducible to be worth citing.
        return sorted(scores, key=lambda s: (-s.score, s.model.name))

    def coverage(self, answers: dict[str, bool]) -> dict:
        """Which tables were actually assessed, for the report to state plainly.

        Presenting a recommendation scored on 10 of 19 questions as a full
        framework assessment would be the kind of thing nobody notices until
        someone asks.
        """
        out = {}
        for table_id, qs in self.tables().items():
            answered = [q.id for q in qs if q.id in answers]
            out[table_id] = {
                "name": qs[0].table_name,
                "source": qs[0].source,
                "answered": len(answered),
                "total": len(qs),
                "assessed": len(answered) > 0,
            }
        return out
