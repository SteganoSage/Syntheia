"""main.py's review triggers and exit codes, and the Ollama wrapper's helpers.

Exit code 1 means "ran, but a human must look", which §6 requires the system to
be able to say. These tests pin what triggers it -- and, just as important,
what must not.
"""

from __future__ import annotations

import pytest

from main import needs_review, read_scenario
from src.models.quality import QualityIssue, QualityReport, QualityResult
from src.models.sdlc import CriterionAnswer, SDLCRecommendation
from src.utils.llm_ollama import extract_json, get_keep_alive


def _rec(**overrides):
    fields = dict(recommendation="Waterfall", ranking=[], criteria=[])
    fields.update(overrides)
    return SDLCRecommendation(**fields)


# ── needs_review ───────────────────────────────────────────────────────

def test_a_clean_run_needs_no_review(make_requirement):
    req = make_requirement()
    state = {"requirements": [req], "grounding": {req.id: {"needs_review": True}},
             "quality_report": None, "sdlc": _rec()}
    # Ungrounded, but it names no regulation -- most requirements are not
    # regulatory, and flagging them all is what made the signal useless.
    assert needs_review(state) == []


def test_a_named_regulation_with_no_support_is_flagged(make_requirement):
    req = make_requirement(applicable_regulations=["RBI Digital Lending Guidelines"])
    state = {"requirements": [req], "grounding": {req.id: {"needs_review": True}}}
    [reason] = needs_review(state)
    assert "RBI Digital Lending Guidelines" in reason
    assert "unverified" in reason


def test_a_critical_quality_issue_is_flagged(make_requirement):
    report = QualityReport(result_list=[QualityResult(
        requirement_id="REQ-001", passed=False,
        issues=[QualityIssue(issue_type="Infeasibility", description="cannot be built",
                             severity="Critical", suggestion="rescope")])])
    reasons = needs_review({"requirements": [make_requirement()], "quality_report": report})
    assert reasons == ["critical quality issue(s): REQ-001/Infeasibility"]


def test_a_framework_tie_is_flagged():
    reasons = needs_review({"sdlc": _rec(tied_with=["V-Shaped", "Waterfall"])})
    assert any("V-Shaped = Waterfall" in r for r in reasons)


def test_an_overruled_answer_is_surfaced_in_words():
    overruled = CriterionAnswer(question_id="PTR-4", answer="not_stated", evidence=[],
                                reasoning="UNSUPPORTED (claimed yes, ...): budget")
    [reason] = needs_review({"sdlc": _rec(criteria=[overruled])})
    assert "funding is settled for the life of the project" in reason
    assert "PTR-4" not in reason, "review notices speak in words, not ids"


# ── read_scenario: exit 2 means "could not run" ────────────────────────

def test_a_missing_scenario_exits_2(tmp_path):
    with pytest.raises(SystemExit) as e:
        read_scenario(str(tmp_path / "nope.txt"))
    assert e.value.code == 2


def test_an_empty_scenario_exits_2(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("   \n", encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        read_scenario(str(empty))
    assert e.value.code == 2


def test_a_directory_is_not_a_scenario(tmp_path):
    with pytest.raises(SystemExit) as e:
        read_scenario(str(tmp_path))
    assert e.value.code == 2


def test_a_scenario_file_is_read_and_stripped(tmp_path):
    f = tmp_path / "s.txt"
    f.write_text("\n  Build a KYC platform.  \n", encoding="utf-8")
    assert read_scenario(str(f)) == "Build a KYC platform."


# ── Ollama wrapper helpers ─────────────────────────────────────────────

def test_keep_alive_minus_one_is_sent_as_a_number(monkeypatch):
    # Problem log 2.3: the string "-1" made Ollama return 400 on every call.
    monkeypatch.setenv("OLLAMA_KEEP_ALIVE", "-1")
    assert get_keep_alive() == -1


def test_keep_alive_durations_stay_strings(monkeypatch):
    monkeypatch.setenv("OLLAMA_KEEP_ALIVE", "30m")
    assert get_keep_alive() == "30m"


@pytest.mark.parametrize("reply, expected", [
    ('{"a": 1}', '{"a": 1}'),
    ('Here is the JSON:\n```json\n{"a": 1}\n```\nHope this helps!', '{"a": 1}'),
    ('Sure! {"a": [1, 2]} -- done', '{"a": [1, 2]}'),
    ('[1, 2, 3]', '[1, 2, 3]'),
])
def test_json_is_salvaged_from_wrapped_replies(reply, expected):
    assert extract_json(reply) == expected


@pytest.mark.parametrize("reply", ["", "no json here", "{broken: ,}"])
def test_unsalvageable_replies_return_none(reply):
    assert extract_json(reply) is None


def test_timed_run_reproduces_invoke_and_times_every_stage():
    from main import _run_timed

    class FakeGraph:
        def stream(self, state, stream_mode):
            assert stream_mode == "updates"
            yield {"a": {"x": 1}}
            yield {"b": {"x": 2, "y": 3}}
            yield {"c": None}

    state, times = _run_timed(FakeGraph(), {"x": 0, "z": 9})
    assert state == {"x": 2, "y": 3, "z": 9}
    assert list(times) == ["a", "b", "c"] and all(t >= 0 for t in times.values())


def test_a_dead_endpoint_is_recognised_through_wrapping():
    from main import _endpoint_failure

    try:
        try:
            raise ConnectionError("HTTPSConnectionPool: Max retries exceeded (Connection refused)")
        except ConnectionError as inner:
            raise RuntimeError("llama3.1:8b failed to produce valid ExtractionDraft") from inner
    except RuntimeError as outer:
        assert _endpoint_failure(outer)


def test_an_ordinary_bug_is_not_mistaken_for_a_dead_endpoint():
    from main import _endpoint_failure

    assert not _endpoint_failure(KeyError("requirements"))
    assert not _endpoint_failure(ValueError("schema mismatch"))
