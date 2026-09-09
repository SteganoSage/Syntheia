"""Merge every classification training candidate into a single training_set.json.

Sources combined (task.md Task 3):
    day7_external_promise_sample.json     60  external PROMISE NFR, single-label
    day7_regulation_grounded_sample.json  23  finance regulation-grounded, multi-label
    synthetic_raw/*.json                  180 Claude-authored, 6 domains, multi-label
    --------------------------------------------------------------------------------
    total                                 263

ground_truth.json is deliberately NOT merged -- it is the held-out evaluation
set. Folding it in would leak eval data into training and invalidate the
fine-tuned-vs-baseline F1 comparison.

Output format matches ground_truth.json: a top-level object with a
"requirements" list of {id, statement, categories, domain, source, verified}.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent

CATEGORIES = [
    "Business", "Stakeholder", "Functional", "Security", "Privacy", "Regulatory",
    "Performance", "Availability", "Usability", "Data-management", "Integration",
    "Audit/Reporting", "Operational",
]

EXTERNAL = HERE / "day7_external_promise_sample.json"
REGULATION = HERE / "day7_regulation_grounded_sample.json"
SYNTHETIC_DIR = HERE / "synthetic_raw"
GROUND_TRUTH = HERE / "ground_truth.json"
OUT = HERE / "training_set.json"

MIN_PER_CATEGORY = 10  # Definition of Done


def load_reqs(path: Path) -> list[dict]:
    return json.loads(path.read_text())["requirements"]


def main() -> None:
    sources: list[tuple[str, Path]] = [
        ("external:PROMISE-NFR", EXTERNAL),
        ("synthetic:regulation-grounded", REGULATION),
    ]
    sources += [
        (f"synthetic:{p.stem}", p)
        for p in sorted(SYNTHETIC_DIR.glob("*.json"))
    ]

    gt_ids = {r["id"] for r in load_reqs(GROUND_TRUTH)}
    gt_statements = {r["statement"].strip().lower() for r in load_reqs(GROUND_TRUTH)}

    merged: list[dict] = []
    seen_ids: set[str] = set()
    seen_statements: set[str] = set()

    for label, path in sources:
        reqs = load_reqs(path)
        kept = 0
        for r in reqs:
            rid = r["id"]
            stmt_key = r["statement"].strip().lower()

            if rid in seen_ids:
                raise SystemExit(f"ID collision: {rid} appears twice (in {path.name})")
            if rid in gt_ids:
                raise SystemExit(f"{rid} from {path.name} collides with a ground_truth.json id")
            if stmt_key in gt_statements:
                raise SystemExit(
                    f"{rid} statement duplicates a ground_truth.json statement -- eval leak:\n"
                    f"  {r['statement']}"
                )
            if stmt_key in seen_statements:
                print(f"  ! skipping {rid} ({path.name}) -- duplicate statement")
                continue

            bad = [c for c in r["categories"] if c not in CATEGORIES]
            if bad:
                raise SystemExit(f"{rid} has unknown categories {bad}")
            if not r["categories"]:
                raise SystemExit(f"{rid} has no categories")

            seen_ids.add(rid)
            seen_statements.add(stmt_key)
            merged.append({
                "id": rid,
                "statement": r["statement"],
                "categories": r["categories"],
                "domain": r.get("domain", ""),
                "source": r.get("source", label),
                "verified": r.get("verified", False),
            })
            kept += 1
        print(f"  {label:38s}  +{kept:3d}  (from {path.name})")

    # keep the file human-scannable: order by numeric requirement id
    merged.sort(key=lambda r: int(r["id"].split("-")[1]))

    # ---- label distribution -------------------------------------------------
    counts = Counter(c for r in merged for c in r["categories"])
    verified = sum(1 for r in merged if r["verified"])
    label_cardinality = sum(len(r["categories"]) for r in merged) / len(merged)

    payload = {
        "dataset": "day7_classification_training_set",
        "purpose": (
            "Merged multi-label training corpus for the Day 9-11 fine-tuned BERT "
            "classifier. Combines external PROMISE-NFR examples, finance "
            "regulation-grounded synthetic requirements, and Claude-authored "
            "synthetic requirements across six domains. SEPARATE from "
            "ground_truth.json, which stays held out for evaluation."
        ),
        "status": "PARTIALLY_UNVERIFIED -- human label pass in progress",
        "categories_taxonomy": CATEGORIES,
        "stats": {
            "total": len(merged),
            "verified": verified,
            "unverified": len(merged) - verified,
            "avg_labels_per_requirement": round(label_cardinality, 2),
            "label_distribution": {c: counts.get(c, 0) for c in CATEGORIES},
        },
        "requirements": merged,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"\n  wrote {OUT.name}: {len(merged)} requirements "
          f"({verified} verified, {len(merged) - verified} to review)")
    print(f"  avg labels/requirement: {label_cardinality:.2f}\n")
    print(f"  {'category':<18}{'count':>6}")
    print(f"  {'-' * 24}")
    low = []
    for c in CATEGORIES:
        n = counts.get(c, 0)
        flag = "  <-- below %d" % MIN_PER_CATEGORY if n < MIN_PER_CATEGORY else ""
        if n < MIN_PER_CATEGORY:
            low.append(c)
        print(f"  {c:<18}{n:>6}{flag}")

    if low:
        print(f"\n  WARNING: {len(low)} categ/ies below {MIN_PER_CATEGORY} examples: {low}")
    else:
        print(f"\n  OK: every category has >= {MIN_PER_CATEGORY} examples")


if __name__ == "__main__":
    main()
