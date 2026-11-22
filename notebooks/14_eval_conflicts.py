"""Evaluate conflict detection on labelled requirement pairs.

    python -m notebooks.14_eval_conflicts              # candidate stage only, no LLM
    python -m notebooks.14_eval_conflicts --judge      # plus the LLM judge
    python -m notebooks.14_eval_conflicts --judge --out evaluation/results/conflicts_8b.json

Two stages, measured separately for the same reason the SDLC eval separates
its layers: a miss at the candidate stage (the judge never saw the pair) wants
a different fix from a miss at the judge (it saw it and got it wrong).

  candidate recall   conflicts/duplicates that reach the judge at all
  candidate noise    unrelated pairs that reach it anyway (cost, not error)
  judge accuracy     per-label, against the gold label, with a confusion matrix

Labels in data/eval/conflict_pairs.yaml are unverified (see the file); treat
the numbers as agreement with a draft until someone has checked them.
Run the judge on the cloud tier -- the local 3B is the dev loop, not the result.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml

from src.agents import conflict
from src.agents.conflict import candidate_pairs, detect_conflicts
from src.models.requirement import Requirement
from src.utils.evidence import quotes_source
from src.utils.similarity import default_similarity

REPO = Path(__file__).resolve().parents[1]
PAIRS = REPO / "data/eval/conflict_pairs.yaml"
LABELS = ["conflict", "duplicate", "overlap", "unrelated"]


def _req(rid: str, text: str) -> Requirement:
    return Requirement(id=rid, statement=text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", action="store_true", help="also run the LLM judge")
    ap.add_argument("--out", type=Path, default=None, help="write results as JSON")
    args = ap.parse_args()

    data = yaml.safe_load(PAIRS.read_text(encoding="utf-8"))
    sim = default_similarity()
    rows = []
    for p in data["pairs"]:
        a, b = _req("REQ-A", p["a"]), _req("REQ-B", p["b"])
        cands = candidate_pairs([a, b], sim)
        row = {"id": p["id"], "gold": p["label"],
               "candidate": bool(cands),
               "signals": cands[0]["signals"] if cands else [],
               "similarity": cands[0]["similarity"] if cands else
               round(sim.cross([a.statement], [b.statement])[0][0], 4)}
        if args.judge and cands:
            # Run the real agent, recording what the judge said along the way.
            # The two can differ by design: a rule-detected clash the judge
            # calls "overlap" is still REPORTED (as possible), so the judge's
            # accuracy is not what a user sees. Both are measured.
            seen = []
            original = conflict.judge
            conflict.judge = lambda x, y: seen.append(original(x, y)) or seen[-1]
            try:
                found = detect_conflicts([a, b], sim)
            finally:
                conflict.judge = original
            if seen:
                v = seen[-1]
                row["judged"] = v.relation
                row["traceable"] = (quotes_source(v.quote_a, a.statement)
                                    and quotes_source(v.quote_b, b.statement))
            row["reported"] = (found[0].kind + ("" if found[0].confirmed else "?")) if found else ""
        rows.append(row)

    print(f"{'id':5s} {'gold':10s} {'sim':>5s}  {'cand':5s} {'judged':10s} {'reported':11s} signals")
    for r in rows:
        judged = r.get("judged", "")
        if judged and not r.get("traceable"):
            judged += "*"
        print(f"{r['id']:5s} {r['gold']:10s} {r['similarity']:5.2f}  "
              f"{'yes' if r['candidate'] else '-':5s} {judged:10s} {r.get('reported', ''):11s} "
              f"{', '.join(r['signals'])}")

    want = [r for r in rows if r["gold"] in ("conflict", "duplicate")]
    noise = [r for r in rows if r["gold"] == "unrelated"]
    recall = sum(r["candidate"] for r in want) / len(want)
    false_cand = sum(r["candidate"] for r in noise) / len(noise) if noise else 0.0
    print(f"\ncandidate recall (conflict+duplicate): {recall:.0%} ({sum(r['candidate'] for r in want)}/{len(want)})")
    print(f"unrelated pairs sent to the judge:     {false_cand:.0%}")

    summary = {"candidate_recall": recall, "unrelated_candidate_rate": false_cand,
               "verified_labels": bool(data.get("verified"))}
    if args.judge:
        judged = [r for r in rows if "judged" in r]
        conf = Counter((r["gold"], r["judged"]) for r in judged)
        acc = sum(r["gold"] == r["judged"] for r in judged) / len(judged) if judged else 0.0
        trace = sum(bool(r["traceable"]) for r in judged) / len(judged) if judged else 0.0
        print(f"\njudge accuracy on candidates: {acc:.0%}   traceable quotes: {trace:.0%}")
        print("confusion (rows gold, cols judged):")
        print("            " + " ".join(f"{l[:9]:>9s}" for l in LABELS))
        for g in LABELS:
            print(f"  {g:10s}" + " ".join(f"{conf[(g, j)]:9d}" for j in LABELS))
        summary.update(judge_accuracy=acc, traceable_rate=trace,
                       confusion={f"{g}->{j}": n for (g, j), n in conf.items()})

        # What the user actually sees. "?" marks an unconfirmed report.
        def caught(kind):
            g = [r for r in rows if r["gold"] == kind]
            return sum(r.get("reported", "").rstrip("?") == kind for r in g), len(g)
        cc, cn = caught("conflict")
        dc, dn = caught("duplicate")
        false = sum(1 for r in rows if r["gold"] in ("overlap", "unrelated")
                    and r.get("reported"))
        unconf = sum(1 for r in rows if r.get("reported", "").endswith("?"))
        print(f"\nREPORTED (what a user sees): conflicts {cc}/{cn}  duplicates {dc}/{dn}  "
              f"false reports on overlap/unrelated {false}  ({unconf} of all reports unconfirmed)")
        summary.update(reported_conflicts=f"{cc}/{cn}", reported_duplicates=f"{dc}/{dn}",
                       false_reports=false, unconfirmed_reports=unconf)
    if not data.get("verified"):
        print("\nNOTE: labels are unverified -- this is agreement with a draft.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
