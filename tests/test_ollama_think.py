"""OLLAMA_THINK: off for a like-for-like comparison, absent for every model that
never had the option -- an unset variable must not change the request."""

from __future__ import annotations

import pytest

from src.utils import llm_ollama


@pytest.fixture
def sent(monkeypatch):
    seen = {}

    class Reply:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": "{}"}

    def fake_post(url, json=None, **kw):
        seen.update(json)
        return Reply()

    monkeypatch.setenv("OLLAMA_STREAM", "0")
    monkeypatch.setattr(llm_ollama.requests, "post", fake_post)
    return seen


@pytest.mark.parametrize("value, expected", [("0", False), ("false", False), ("1", True)])
def test_think_is_sent_when_set(monkeypatch, sent, value, expected):
    monkeypatch.setenv("OLLAMA_THINK", value)
    llm_ollama.generate("s", "u")
    assert sent["think"] is expected


def test_unset_sends_nothing(monkeypatch, sent):
    monkeypatch.delenv("OLLAMA_THINK", raising=False)
    llm_ollama.generate("s", "u")
    assert "think" not in sent
