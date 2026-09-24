#!/usr/bin/env python3
"""Syntheia — requirements and SDLC recommendation from a problem scenario.

The entry point the plan asks for on Days 18–19. Both deliverables in one
command, and **no third-party API in the path**: extraction, quality and SDLC
run on our own model through Ollama (local 3B or cloud 8B, selected by
`OLLAMA_HOST`), and classification runs the fine-tuned RoBERTa.

    python main.py                                # interactive application
    python main.py scenario.txt                   # one-shot; still asks follow-ups
    python main.py scenario.txt --no-interview    # one-shot, no questions
    python main.py data/test_cases/kyc_onboarding/scenario_01.txt
    echo "Build a payments reconciliation engine..." | python main.py -
    python main.py scenario.txt --no-rag          # ungrounded control arm
    python main.py scenario.txt --json run.json   # machine-readable

`python -m src.graph` still works and is equivalent; it delegates here so the
two cannot drift apart.

Exit codes, so this composes in a shell script:

    0  ran, and nothing needs a human
    1  ran, but something is flagged for review -- an ungrounded regulation
       claim, a critical quality issue, or a framework tie
    2  could not run (bad scenario, no LLM reachable)

Code 1 is not an error. It is the pipeline saying it is unsure, which §6
requires it to be able to say. A caller that treats "unsure" as success is the
reason `in_taxonomy_rate` read 100% for a model spraying labels at everything.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

DEFAULT_SCENARIO = Path("data/test_cases/loan_processing/scenario_01.txt")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="syntheia",
        description="Generate classified, grounded requirements from a problem "
                    "scenario, then recommend an SDLC with a justification.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Set OLLAMA_HOST to point at the cloud tier; unset it for local.",
    )
    p.add_argument(
        "scenario", nargs="?", type=str, default=None, metavar="SCENARIO",
        help="path to a scenario file, or '-' to read stdin. Omit both this "
             "and --scenario to start the interactive application.",
    )
    p.add_argument(
        "--demo", action="store_true",
        help=f"one-shot run of the built-in scenario ({DEFAULT_SCENARIO})",
    )
    # Kept as well as the positional: every existing command line, the runbook
    # and the eval loop all pass --scenario, and silently breaking them to gain
    # a nicer signature is not worth it.
    p.add_argument("--scenario", dest="scenario_opt", type=str, default=None,
                   help=argparse.SUPPRESS)

    out = p.add_argument_group("output")
    out.add_argument("--json", type=Path, default=None, metavar="PATH",
                     help="also write the run as JSON (read by "
                          "notebooks/11_eval_sdlc.py --runs)")
    out.add_argument("--detail", action="store_true",
                     help="also print the framework working -- every score, "
                          "answer and coverage figure. Off by default: it is "
                          "how the decision is computed, not what it is")
    out.add_argument("--quiet", action="store_true",
                     help="suppress the report; useful with --json")

    stages = p.add_argument_group("stages")
    stages.add_argument("--no-rag", action="store_true",
                        help="skip retrieval and generate from the model's own "
                             "knowledge -- the control arm for the "
                             "grounded-vs-ungrounded comparison")
    stages.add_argument("--no-interview", action="store_true",
                        help="never ask follow-up questions, even at a terminal "
                             "-- for reproducible runs and timing measurements")
    stages.add_argument("--no-sdlc", action="store_true",
                        help="stop after Phase 1 -- faster when iterating on "
                             "extraction")
    stages.add_argument("--classifier", default="auto",
                        choices=["auto", "bert", "few_shot", "zero_shot"],
                        help="auto uses the fine-tuned model when one is "
                             "trained, else the few-shot LLM baseline")
    return p


def _die(message: str) -> None:
    """Exit 2 -- "could not run", distinct from 1 which means "ran, needs review".

    `sys.exit("text")` prints the message but exits 1, which would collide with
    the review code and make the two indistinguishable to a caller.
    """
    print(f"syntheia: {message}", file=sys.stderr)
    raise SystemExit(2)


def read_scenario(arg: str | None) -> str:
    """Resolve the scenario text, or exit 2 saying why.

    Exits rather than raising: a traceback for "that file does not exist" tells
    the reader nothing they did not already know.
    """
    if arg == "-":
        text = sys.stdin.read().strip()
        if not text:
            _die("stdin was empty")
        return text

    path = Path(arg) if arg else DEFAULT_SCENARIO
    if not path.exists():
        _die(f"no such scenario file: {path}")
    if path.is_dir():
        _die(f"{path} is a directory, not a scenario file")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        _die(f"{path} is empty")
    return text


def needs_review(state: dict) -> list[str]:
    """Why this run should not be taken at face value. Drives exit code 1."""
    reasons = []

    # `needs_review` from the retriever means "no regulation cleared the
    # confidence floor", which is not by itself a problem -- most requirements
    # are not regulatory. It is a problem when the model NAMED a regulation and
    # the corpus cannot corroborate it, which is the §6 escalation case. The
    # retriever reports the fact; deciding what to escalate is this function's
    # job, and conflating the two flagged every testing requirement.
    # In words, not ids. "PTR-5" and "REQ-001" are internal labels; a reader
    # seeing them has to go and look them up, which for a review notice is
    # exactly backwards -- the point is to say plainly what is unsafe to trust.
    by_req = {r.id: r for r in state.get("requirements", [])}
    ungrounded = sorted(rid for rid, g in (state.get("grounding") or {}).items()
                        if g.get("needs_review") and by_req.get(rid)
                        and by_req[rid].applicable_regulations)
    for rid in ungrounded:
        req = by_req[rid]
        named = ", ".join(req.applicable_regulations)
        reasons.append(
            f"\"{req.statement}\" claims {named} applies, but nothing in the "
            f"knowledge base supports that — the citation is unverified.")

    report = state.get("quality_report")
    if report:
        critical = [f"{r.requirement_id}/{i.issue_type}"
                    for r in report.result_list for i in r.issues
                    if str(i.severity).lower() == "critical"]
        if critical:
            reasons.append(f"critical quality issue(s): {', '.join(critical)}")

    rec = state.get("sdlc")
    if rec and rec.tied_with:
        reasons.append(
            f"the framework does not separate {' = '.join(rec.tied_with)}; "
            f"the reported recommendation is first by name, not by fit")
    if rec:
        # An answer the gate overruled is not an abstention -- the model
        # committed and could not support it. Worth surfacing, because it
        # silently shrinks the scoring denominator.
        overruled = [a for a in rec.criteria
                     if a.reasoning.startswith("UNSUPPORTED")]
        if overruled:
            from src.sdlc.framework import Framework

            fw = Framework.load()
            for a in overruled:
                q = fw.question(a.question_id)
                # `says_yes` is already a statement -- "the product will be
                # running for years" -- so it reads after "whether". The
                # question text does not: it starts "Is the product...", giving
                # "whether is the product...".
                subject = (q.says_yes or q.text.rstrip("?")) if q else a.question_id
                reasons.append(
                    f"could not establish whether {subject} — the model gave an "
                    f"answer but could not point at anything in the scenario "
                    f"that says so, so it was left out of the scoring.")
    return reasons


def write_json(path: Path, scenario_arg: str | None, text: str, state: dict) -> None:
    import json

    rec = state.get("sdlc")
    path.parent.mkdir(parents=True, exist_ok=True)
    src = Path(scenario_arg) if scenario_arg and scenario_arg != "-" else None
    path.write_text(json.dumps({
        "scenario": str(src) if src else "<stdin>",
        # The eval harness keys on this to find the matching expected.yaml.
        "scenario_name": src.parent.name if src else "<stdin>",
        "scenario_text": text,
        "requirements": [r.model_dump() for r in state.get("requirements", [])],
        "grounding": state.get("grounding", {}),
        "quality_report": (state["quality_report"].model_dump()
                           if state.get("quality_report") else None),
        "sdlc": rec.model_dump() if rec else None,
        "needs_review": needs_review(state),
    }, indent=2, default=str), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scenario_arg = args.scenario_opt or args.scenario

    # No scenario named and no --demo: this is a person, not a script. Start the
    # application rather than silently analysing a default file they did not ask
    # for -- which is what the old behaviour did, and it was confusing.
    if scenario_arg is None and not args.demo:
        if not sys.stdin.isatty():
            _die("no scenario given. Pass a file, or '-' to read stdin.")
        from src.cli.app import App

        return App(use_rag=not args.no_rag, run_sdlc=not args.no_sdlc,
                   classifier=args.classifier).run()

    text = read_scenario(scenario_arg)

    from src.agents.classification import resolve_model_dir
    from src.graph import build_pipeline
    from src.utils.warmup import warm

    if not args.quiet:
        model_dir = resolve_model_dir()
        if args.classifier in ("auto", "bert") and model_dir:
            print(f"Classifier: fine-tuned ({model_dir.name})")
        elif args.classifier in ("auto", "bert"):
            print("Classifier: no trained model found, falling back to few-shot LLM")
        else:
            print(f"Classifier: {args.classifier} LLM baseline")

    # Load the classifier and pin the LLM before the clock starts. On a cold
    # process this is ~96% of a run's wall-clock (Day 13 measurement) and it is
    # one-off, not per-request -- charging it to startup is the difference
    # between a 10-minute demo and a 1-minute one.
    warm()

    if not args.quiet:
        print(f"Grounding:  {'OFF (--no-rag)' if args.no_rag else 'retrieval-augmented'}")

    t0 = time.perf_counter()
    state = build_pipeline().invoke({
        "text": text,
        "requirements": [],
        "quality_report": None,
        "classifier": args.classifier,
        "use_rag": not args.no_rag,
        "context": [],
        "grounding": {},
        "sdlc": None,
        "run_sdlc": not args.no_sdlc,
        # Follow-up questions only when a person is actually there to answer.
        #
        # stdin ONLY. The first version also required stdout to be a tty, which
        # is the wrong test and silently disabled the whole feature: redirecting
        # or wrapping output does not stop anyone typing an answer -- only a
        # piped stdin does. `--quiet` and `--no-interview` are explicit refusals.
        "interactive": (sys.stdin.isatty()
                        and not args.quiet and not args.no_interview),
    })
    elapsed = time.perf_counter() - t0

    if not args.quiet:
        from src.cli import render

        render.print_run(state, elapsed, detail=args.detail)

    if args.json:
        write_json(args.json, scenario_arg, text, state)
        if not args.quiet:
            print(f"Wrote {args.json}")

    reasons = needs_review(state)
    if not args.quiet:
        from src.cli import render

        render.print_review(reasons)
    return 1 if reasons else 0


if __name__ == "__main__":
    raise SystemExit(main())
