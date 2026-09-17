"""Score the SDLC engine against written-down expectations.

Until now every comparison was me reading five screens of terminal output and
counting. That does not survive a change: after the evidence gate and the tie
reporting there was no way to say whether the engine got better, only that the
output looked different. So the expectations live in
`data/test_cases/<scenario>/expected.yaml` and this reads them.

Two modes, because there are two separate things that can be wrong:

    --framework          Score the EXPECTED answers through the framework.
                         No LLM runs. Answers the question "if the model
                         inferred perfectly, would the framework pick the right
                         SDLC?" -- which is about the tables, not the model.

    --runs DIR           Compare saved pipeline runs (src.graph --json) against
                         the expected answers. Answers "how close did the model
                         get?", per question, so a regression names its question
                         instead of showing up as a different recommendation.

Run the framework mode first. If the framework cannot reach the right answer
from perfect inputs, no amount of prompt work on the inference step will help,
and the problem is the tables or the expectations -- not the LLM.

    python -m notebooks.11_eval_sdlc --framework
    python -m notebooks.11_eval_sdlc --runs evaluation/results/sdlc_runs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.sdlc.framework import Framework

REPO = Path(__file__).resolve().parents[1]
CASES = REPO / "data/test_cases"


# YAML 1.1 reads a bare `yes` as the boolean True, so the files come back with
# a mix of bools and the string "not_stated". Normalised here rather than by
# quoting every value in the files, which reads worse and is easy to forget.
_ANSWER = {True: "yes", False: "no", "yes": "yes", "no": "no",
           "not_stated": "not_stated"}


def load_expected() -> list[dict]:
    out = []
    for path in sorted(CASES.glob("*/expected.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        answers = {}
        for qid, raw in (data.get("answers") or {}).items():
            if raw not in _ANSWER:
                raise SystemExit(f"{path}: {qid} is {raw!r}; "
                                 f"expected yes, no or not_stated")
            answers[qid] = _ANSWER[raw]
        data["answers"] = answers
        data["path"] = path
        out.append(data)
    if not out:
        raise SystemExit(f"no expected.yaml under {CASES}")
    return out


def as_bools(answers: dict) -> dict[str, bool]:
    """Drop not_stated -- unanswered questions are not scored, for anyone."""
    return {k: v == "yes" for k, v in answers.items() if v in ("yes", "no")}


def check_ids(fw: Framework, cases: list[dict]) -> None:
    """Every question answered must exist, and every question must be answered.

    Same reasoning as Framework.load being strict about missing model columns: a
    typo'd id would silently shrink the denominator and the numbers would all
    still look plausible.
    """
    known = {q.id for q in fw.questions}
    for case in cases:
        got = set(case["answers"])
        if got != known:
            raise SystemExit(
                f"{case['path']}: "
                + (f"missing {sorted(known - got)} " if known - got else "")
                + (f"unknown {sorted(got - known)}" if got - known else "")
            )


# ── Mode 1: is the framework able to get there at all? ─────────────────

def eval_framework(fw: Framework, cases: list[dict]) -> int:
    print("\nScoring the EXPECTED answers through the framework.")
    print("No LLM involved: this tests the tables and the expectations only.\n")

    failures = 0
    for case in cases:
        answers = as_bools(case["answers"])
        scores = fw.score(answers)
        top = scores[0]
        tied = [s.model.name for s in scores if abs(s.score - top.score) < 1e-9]
        wanted = case["expected_models"]

        hit = any(name in wanted for name in tied)
        # A tie that contains the right answer is not a pass. The framework was
        # asked to discriminate and did not; bank_migration's V-Shaped came out
        # of an alphabetical tie-break against Waterfall and looked like a win.
        clean = hit and len(tied) == 1
        mark = "PASS " if clean else ("TIE  " if hit else "FAIL ")
        failures += 0 if clean else 1

        print(f"  {mark} {case['scenario']:<18} want {'/'.join(wanted):<18} "
              f"got {' = '.join(tied)}  ({top.score:.0%}, "
              f"{len(top.matched)}/{top.answered} of {len(answers)} answered)")
        if not clean:
            for s in scores[:4]:
                flag = "<-- wanted" if s.model.name in wanted else ""
                print(f"           {s.model.name:<14}{s.score:>6.0%}  "
                      f"{len(s.matched)}/{s.answered} {flag}")
        if not case.get("verified"):
            print(f"           (answers unverified -- {case['path'].parent.name}"
                  f"/expected.yaml)")
    return failures


# ── Mode 2: how close did the model get? ───────────────────────────────

def eval_runs(fw: Framework, cases: list[dict], runs_dir: Path) -> int:
    by_name = {c["scenario"]: c for c in cases}
    run_files = sorted(runs_dir.glob("*.json"))
    if not run_files:
        raise SystemExit(f"no *.json in {runs_dir} -- produce them with "
                         f"`python -m src.graph --scenario ... --json {runs_dir}/<name>.json`")

    tally = {"right": 0, "wrong": 0, "missed": 0, "unsupported": 0, "asked": 0}
    failures = 0

    for path in run_files:
        run = json.loads(path.read_text(encoding="utf-8"))
        name = run.get("scenario_name", path.stem)
        case = by_name.get(name)
        if not case or not run.get("sdlc"):
            print(f"  SKIP  {name}: no expected.yaml or no sdlc block")
            continue

        rec = run["sdlc"]
        got = {a["question_id"]: a["answer"] for a in rec["criteria"]}
        want = case["answers"]

        right = wrong = abstained = overreached = 0
        detail = []
        for qid, expected in want.items():
            actual = got.get(qid, "not_stated")
            if expected == actual:
                right += 1
            elif actual == "not_stated":
                abstained += 1          # should have answered, didn't
                detail.append((qid, expected, actual))
            elif expected == "not_stated":
                overreached += 1        # answered from nothing -- the gate's job
                detail.append((qid, expected, actual))
            else:
                wrong += 1              # answered, and wrong
                detail.append((qid, expected, actual))

        tally["right"] += right
        tally["wrong"] += wrong
        tally["missed"] += abstained
        tally["unsupported"] += overreached
        tally["asked"] += sum(1 for v in want.values() if v != "not_stated")

        wanted = case["expected_models"]
        tied = rec.get("tied_with") or [rec["recommendation"]]
        hit = any(n in wanted for n in tied)
        clean = hit and len(tied) == 1
        failures += 0 if clean else 1
        mark = "PASS " if clean else ("TIE  " if hit else "FAIL ")

        print(f"\n  {mark} {name:<18} want {'/'.join(wanted):<18} "
              f"got {' = '.join(tied)}")
        print(f"         answers: {right}/{len(want)} match  "
              f"({wrong} wrong, {abstained} missed, {overreached} unsupported)")
        for qid, exp, act in detail:
            q = fw.question(qid)
            print(f"           {qid:<8} want {exp:<11} got {act:<11} {q.text[:46] if q else ''}")

    # Three failure modes, and they want different fixes. `wrong` is an
    # inference problem -- a better prompt or a fine-tune. `missed` is the
    # evidence gate or the abstention prompt being too strict. `unsupported` is
    # fabrication, and should now be near zero.
    print(f"\n  Totals over {len(run_files)} runs:")
    print(f"    {tally['right']:>3} correct")
    print(f"    {tally['wrong']:>3} answered wrongly        (inference quality)")
    print(f"    {tally['missed']:>3} missed                  (scenario says it, model abstained)")
    print(f"    {tally['unsupported']:>3} unsupported             (answered from nothing)")
    print(f"    {tally['asked']:>3} questions the scenarios actually answer")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--framework", action="store_true",
                    help="score the expected answers through the tables (no LLM)")
    ap.add_argument("--runs", type=Path, default=None,
                    help="directory of JSON runs from `src.graph --json`")
    args = ap.parse_args()
    if not args.framework and not args.runs:
        args.framework = True

    fw = Framework.load()
    cases = load_expected()
    check_ids(fw, cases)
    unverified = [c["scenario"] for c in cases if not c.get("verified")]

    failures = 0
    if args.framework:
        failures += eval_framework(fw, cases)
    if args.runs:
        failures += eval_runs(fw, cases, args.runs)

    if unverified:
        print(f"\n  NOTE: expected answers are unverified for "
              f"{', '.join(unverified)}.")
        print(f"  Recommendation-level results above are real; answer-level "
              f"results are a comparison against a draft.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
