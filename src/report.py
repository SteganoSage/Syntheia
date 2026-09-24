"""Render a pipeline run to the terminal.

Lifted out of `src/graph.py`'s __main__ so that `main.py` and `python -m
src.graph` share one renderer instead of drifting into two. Presentation only:
nothing here decides anything, and nothing here calls a model.

The one piece of judgement in this file is what gets *emphasised*. A run's
output is long, and the parts a reader must not miss are the ones that say the
system is unsure -- an ungrounded citation, a tie the framework could not break,
an answer the evidence gate overruled. Those are printed loudly on purpose. A
report that reads equally confident everywhere is the failure mode this project
keeps rediscovering.
"""

from __future__ import annotations

from src.models.quality import QualityReport
from src.models.requirement import Requirement
from src.models.sdlc import SDLCRecommendation

RULE = "=" * 60


def print_requirements(requirements: list[Requirement],
                       grounding: dict,
                       elapsed: float | None = None) -> None:
    timing = f" in {elapsed:.1f}s (warm)" if elapsed is not None else ""
    print(f"Extracted {len(requirements)} requirements{timing}:\n")

    for req in requirements:
        print(f"[{req.id}] ({req.priority}) {req.statement}")
        print(f"    Categories:       {req.category}")
        print(f"    Stakeholder:      {req.source_stakeholder}")
        print(f"    Justification:    {req.business_justification}")
        print(f"    Acceptance:       {req.acceptance_criteria}")

        g = (grounding or {}).get(req.id)
        if g and g["citations"]:
            print("    Regulations:      (grounded)")
            for c in g["citations"]:
                print(f"        [{c['confidence']:.2f}] {c['citation']}")
        elif g and g["needs_review"]:
            # Loud on purpose. The model asserted a regulation applies and the
            # knowledge base cannot corroborate it -- which is exactly the
            # escalation §6 asks for, and useless if it reads like a footnote.
            print(f"    Regulations:      {req.applicable_regulations}")
            print(f"        !! no regulation in the knowledge base supports this "
                  f"(best match {g['confidence']:.2f}) -- needs human review")
        else:
            print(f"    Regulations:      {req.applicable_regulations}")

        if g and g.get("precedents"):
            # Not regulations -- how comparable institutions specified the same
            # thing. Kept visually separate so a tender is never mistaken for a
            # binding source.
            print("    Precedent:        (how other institutions specified this)")
            for c in g["precedents"][:2]:
                print(f"        [{c['confidence']:.2f}] {c['citation'][:96]}")

        print(f"    Risk:             {req.risk_level}")
        print(f"    Confidence:       {req.confidence_score}")
        print()


def print_quality(report: QualityReport | None) -> None:
    if not report:
        return
    passed = sum(1 for r in report.result_list if r.passed)
    failed = len(report.result_list) - passed
    print(RULE)
    print(f"QUALITY REPORT: {passed} passed, {failed} flagged\n")
    for qr in report.result_list:
        print(f"  {'✅' if qr.passed else '❌'} [{qr.requirement_id}]")
        for issue in qr.issues:
            print(f"      ⚠️ {issue.issue_type} ({issue.severity}): {issue.description}")
            print(f"         → {issue.suggestion}")
    print()


def print_sdlc(rec: SDLCRecommendation | None) -> None:
    if not rec:
        return

    print(RULE)
    derived = "  [derived column -- not in the supplied framework]" if any(
        m.is_derived and m.name == rec.recommendation for m in rec.ranking) else ""
    print(f"SDLC RECOMMENDATION: {rec.recommendation}{derived}")
    if rec.tied_with:
        print(f"  ** TIED: {' = '.join(rec.tied_with)} -- the framework does "
              f"not separate them.")
        print("     Listed first by name, which is not a preference. Answer "
              "an unassessed question to break it.")
    print(f"{RULE}\n")

    print("  Framework scoring:")
    for s in rec.ranking:
        mark = " *" if s.name == rec.recommendation else "  "
        d = "  (derived)" if s.is_derived else ""
        print(f"   {mark} {s.name:<14}{s.score:>6.0%}  "
              f"{len(s.matched)}/{len(s.matched) + len(s.missed)}{d}")

    print("\n  Inferred from the requirements:")
    for a in rec.criteria:
        ev = ", ".join(e[:44] for e in a.evidence) if a.evidence else "nothing cited"
        label = {"yes": "YES", "no": "NO", "not_stated": "--"}[a.answer]
        # An answer the evidence gate rejected reads as an abstention in the
        # score, but it is not one -- the model did commit, and a reader
        # checking the framework should see that it was overruled.
        if a.reasoning.startswith("UNSUPPORTED"):
            ev = a.reasoning.split("):")[0] + ")"
        print(f"    {a.question_id:<7} {label:<4} [{ev}]")

    print("\n  Framework coverage:")
    for c in rec.coverage.values():
        print(f"    {c.name:<22} {c.answered}/{c.total:<3} "
              f"{'assessed' if c.assessed else 'NOT ASSESSED'} ({c.source})")

    if rec.cross_check_ranking:
        agree = "AGREES" if rec.cross_check_agrees else "DISAGREES"
        print(f"\n  Cross-check ({agree}): the model ranked, without seeing scores,")
        print(f"    {' > '.join(rec.cross_check_ranking[:4])}")

    for j in rec.justifications:
        print(f"\n  --- {j.model} ---")
        for b in j.why_it_fits:
            print(f"    + {b}")
        for a in j.advantages:
            print(f"    * {a.advantage}: {a.relevance}")
        if j.gives_up:
            print(f"    - gives up: {j.gives_up}")
    if rec.summary:
        print(f"\n  {rec.summary}")
    if rec.shortcut:
        print(f"\n  IN ONE LINE: {rec.shortcut}")
    print()


def print_run(state: dict, elapsed: float | None = None) -> None:
    """Everything, in pipeline order."""
    print_requirements(state.get("requirements", []),
                       state.get("grounding", {}), elapsed)
    print_quality(state.get("quality_report"))
    print_sdlc(state.get("sdlc"))
