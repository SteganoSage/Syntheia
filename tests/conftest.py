"""Shared setup for the test suite.

Every test here is deterministic and offline. Nothing calls an LLM, nothing
loads RoBERTa, nothing embeds text. What is covered is the part of the system
that decides things in code -- scoring, the evidence gate, the chunker, the
citation checks, the quality filters, the review triggers -- because those are
where a silent regression changes results without changing how the output
looks. Most entries in docs/problem_log.md are exactly that kind of bug.

    ~/venvs/syntheia/bin/python -m pytest tests -q
"""

from __future__ import annotations

import os

# Must run before anything imports src.utils.llm. Without an explicit provider
# the gateway probes Ollama over HTTP at import time, which makes the suite
# slow when the daemon is up and changes the code path when it is down. Naming
# the provider skips the probe; no test makes a call through it.
os.environ.setdefault("LLM_PROVIDER", "ollama")

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session")
def framework():
    from src.sdlc.framework import Framework

    return Framework.load()


@pytest.fixture
def make_requirement():
    """A valid Requirement with sensible defaults; override any field."""
    from src.models.requirement import Requirement

    def _make(**overrides):
        fields = dict(
            id="REQ-001",
            statement="The system shall verify the customer's PAN against NSDL.",
            source_stakeholder="Compliance officer",
            business_justification="Mandated identity verification.",
            priority="High",
            dependencies=[],
            assumptions="",
            acceptance_criteria="",
            applicable_regulations=[],
            risk_level="Medium",
            confidence_score=0.9,
            approval_status="Pending",
            category=[],
        )
        fields.update(overrides)
        return Requirement(**fields)

    return _make
