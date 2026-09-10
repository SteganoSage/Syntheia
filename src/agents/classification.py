"""Classification agent — assigns multi-label categories to requirements.

Day 5:    two LLM-based baselines (zero-shot and few-shot).
Day 9-11: fine-tuned BERT/RoBERTa classifier, now the pipeline default.

The LLM baselines are deliberately kept, not deleted. Every fine-tuned model in
this project is benchmarked against them, and `classify_few_shot` is still the
fallback when no trained model is on disk.

On the held-out ground_truth.json (32 requirements):

    classifier            micro-F1   latency/req
    zero-shot (Gemini)      80.7%        23.20 s
    few-shot  (Gemini)      84.0%        30.55 s
    RoBERTa (fine-tuned)    80.2%        0.021 s

The fine-tuned model gives up 3.8 points of micro-F1 for roughly 1,450x lower
latency, which is the trade the pipeline wants: classification runs on every
requirement of every scenario, so 30 s per requirement does not scale.

The 13-category taxonomy:
  Business, Stakeholder, Functional, Security, Privacy, Regulatory,
  Performance, Availability, Usability, Data-management, Integration,
  Audit/Reporting, Operational
"""

import os
from pathlib import Path

from pydantic import BaseModel, Field

from src.models.requirement import Requirement
from src.utils.llm import call_structured

REPO = Path(__file__).resolve().parents[2]

# Searched in order; the first directory holding a trained model wins. RoBERTa
# leads because it scores higher (80.2% vs 78.0% on ground truth). Override with
# SYNTHEIA_CLASSIFIER_DIR to pin a specific run.
MODEL_SEARCH_PATH = [
    REPO / "models/roberta_classifier",
    REPO / "models/bert_classifier",
]

ALL_CATEGORIES = [
    "Business",
    "Stakeholder",
    "Functional",
    "Security",
    "Privacy",
    "Regulatory",
    "Performance",
    "Availability",
    "Usability",
    "Data-management",
    "Integration",
    "Audit/Reporting",
    "Operational",
]


class ClassificationResult(BaseModel):
    categories: list[str] = Field(
        description="One or more categories from the 13-category taxonomy"
    )


# ── Prompts ────────────────────────────────────────────────────────────

ZERO_SHOT_PROMPT = f"""\
You are a software requirements classifier for financial systems.

Given a software requirement statement, classify it into ONE OR MORE
categories from the following list (multi-label):

{chr(10).join(f'  - {c}' for c in ALL_CATEGORIES)}

Rules:
- Return ONLY categories from the list above — no new categories.
- A requirement may belong to multiple categories.
- Be precise: only assign a category if the requirement clearly relates to it.
"""

FEW_SHOT_PROMPT = f"""\
You are a software requirements classifier for financial systems.

Given a software requirement statement, classify it into ONE OR MORE
categories from the following list (multi-label):

{chr(10).join(f'  - {c}' for c in ALL_CATEGORIES)}

Rules:
- Return ONLY categories from the list above — no new categories.
- A requirement may belong to multiple categories.
- Be precise: only assign a category if the requirement clearly relates to it.

Here are examples of correct classification:

Example 1:
Requirement: "The system shall encrypt all applicant PII, including PAN and Aadhaar numbers, at rest using AES-256 encryption."
Categories: ["Security", "Privacy", "Data-management"]

Example 2:
Requirement: "The system shall route loan applications exceeding ₹50 lakhs to a senior loan officer for manual approval."
Categories: ["Business", "Functional", "Stakeholder"]

Example 3:
Requirement: "The system shall generate a monthly compliance report listing all loan applications that required manual override of automated approval rules."
Categories: ["Audit/Reporting", "Regulatory"]

Example 4:
Requirement: "The system shall support at least 500 concurrent loan application submissions without response times exceeding 5 seconds."
Categories: ["Performance"]

Example 5:
Requirement: "The system shall integrate with the UIDAI Aadhaar authentication API to perform real-time identity verification."
Categories: ["Integration", "Regulatory"]
"""


# ── Classification functions ───────────────────────────────────────────

def classify_zero_shot(statement: str) -> list[str]:
    """Classify a requirement using zero-shot prompting (no examples)."""
    result = call_structured(
        system_prompt=ZERO_SHOT_PROMPT,
        user_prompt=f'Requirement: "{statement}"',
        response_model=ClassificationResult,
    )
    # Filter to valid categories only
    return [c for c in result.categories if c in ALL_CATEGORIES]


def classify_few_shot(statement: str) -> list[str]:
    """Classify a requirement using few-shot prompting (with examples)."""
    result = call_structured(
        system_prompt=FEW_SHOT_PROMPT,
        user_prompt=f'Requirement: "{statement}"',
        response_model=ClassificationResult,
    )
    return [c for c in result.categories if c in ALL_CATEGORIES]


# ── Fine-tuned classifier ──────────────────────────────────────────────

_classifier = None          # cached across calls; loading costs ~2 s
_classifier_dir = None


def resolve_model_dir() -> Path | None:
    """First trained model on the search path, or None if there is none.

    SYNTHEIA_CLASSIFIER_DIR pins a specific directory, which is how you point
    the pipeline at an experiment without editing code.
    """
    override = os.environ.get("SYNTHEIA_CLASSIFIER_DIR")
    candidates = [Path(override)] if override else MODEL_SEARCH_PATH
    for path in candidates:
        if (path / "config.json").exists():
            return path
    return None


def get_classifier(model_dir: str | Path | None = None):
    """Load and cache the fine-tuned classifier. Returns None if none exists.

    Loading is lazy rather than at import time: `src.agents.classification` is
    imported by the baseline scripts too, and they should not pay for ~440 MB
    of weights they never touch.
    """
    global _classifier, _classifier_dir

    target = Path(model_dir) if model_dir else resolve_model_dir()
    if target is None:
        return None
    if _classifier is not None and _classifier_dir == target:
        return _classifier

    from src.models.multilabel import MultiLabelClassifier

    _classifier = MultiLabelClassifier(target)
    _classifier_dir = target
    return _classifier


def classify_bert(statement: str, model_dir: str | Path | None = None) -> list[str]:
    """Classify one requirement with the fine-tuned model.

    Named classify_bert for continuity with the Day 9-11 task list; it serves
    whichever architecture is on disk, RoBERTa included.
    """
    clf = get_classifier(model_dir)
    if clf is None:
        raise SystemExit(
            "No fine-tuned classifier found. Train one with "
            "`python -m notebooks.05_train_bert`, or set SYNTHEIA_CLASSIFIER_DIR."
        )
    return [c for c in clf.predict_one(statement) if c in ALL_CATEGORIES]


def classify_batch(statements: list[str], model_dir: str | Path | None = None
                   ) -> list[list[str]]:
    """Classify many statements in one forward pass.

    Batching is ~6x cheaper per requirement than looping classify_bert
    (3.6 ms vs 21 ms), which is why the pipeline node uses this.
    """
    clf = get_classifier(model_dir)
    if clf is None:
        raise SystemExit("No fine-tuned classifier found.")
    return [
        [c for c in cats if c in ALL_CATEGORIES]
        for cats in clf.predict(statements)
    ]


# ── Pipeline entry point ───────────────────────────────────────────────

def classify_requirements(
    requirements: list[Requirement],
    method: str = "auto",
) -> list[Requirement]:
    """Classify a batch of requirements. Used by the LangGraph pipeline.

    method:
        "auto"      fine-tuned model when one is on disk, else few-shot LLM
        "bert"      fine-tuned model, error if none is trained
        "few_shot"  LLM few-shot baseline
        "zero_shot" LLM zero-shot baseline

    Sets `category` on each requirement in place and returns the list.
    """
    if not requirements:
        return requirements

    if method == "auto":
        method = "bert" if resolve_model_dir() else "few_shot"

    if method == "bert":
        statements = [r.statement for r in requirements]
        for req, cats in zip(requirements, classify_batch(statements)):
            req.category = cats
        return requirements

    per_statement = {"few_shot": classify_few_shot, "zero_shot": classify_zero_shot}
    if method not in per_statement:
        raise ValueError(
            f"unknown method {method!r}; expected auto, bert, few_shot or zero_shot"
        )
    for req in requirements:
        req.category = per_statement[method](req.statement)
    return requirements
