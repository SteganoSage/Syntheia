"""`main.py --export DIR` in the interactive app: every finished scenario is
written without a command, each to its own numbered folder."""

from __future__ import annotations

import json

import pytest

from src.cli.app import App


@pytest.fixture
def app(tmp_path, make_requirement, monkeypatch):
    written = []

    def fake_generate(payload, folder, **kw):
        folder.mkdir(parents=True, exist_ok=True)
        p = folder / "srs.md"
        p.write_text("# SRS")
        written.append((folder.name, kw["formats"]))
        return {"md": p}

    monkeypatch.setattr("src.generators.srs.generate_srs", fake_generate)
    a = App(export_dir=tmp_path / "runs")
    a.written = written
    a.state = lambda text: {"text": text, "requirements": [make_requirement(id="REQ-001")]}
    return a


def test_each_scenario_gets_its_own_numbered_folder(app, tmp_path):
    app.last_state = app.state("Build a digital customer onboarding platform for a neo-bank.")
    app._auto_export()
    app.last_state = app.state("A fintech startup wants a loan app.")
    app._auto_export()
    assert [w[0] for w in app.written] == ["01-build-a-digital-customer-onboarding",
                                           "02-a-fintech-startup-wants-a"]
    run = json.loads((tmp_path / "runs/01-build-a-digital-customer-onboarding/run.json").read_text())
    assert run["requirements"][0]["id"] == "REQ-001"


def test_nothing_is_written_without_requirements_or_without_the_flag(app, make_requirement):
    app.last_state = {"text": "x", "requirements": []}
    app._auto_export()
    app.export_dir = None
    app.last_state = app.state("y")
    app._auto_export()
    assert app.written == []


def test_a_failed_export_does_not_end_the_session(app, monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("tunnel gone")

    monkeypatch.setattr("src.generators.srs.generate_srs", boom)
    app.last_state = app.state("Build a platform.")
    app._auto_export()                     # must not raise
