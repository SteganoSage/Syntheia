"""Merge every classification training candidate into a single training_set.json.

Sources combined:
    day7_finance_specific.json             80  finance-specific, multi-label
    day7_regulation_grounded_sample.json   23  finance regulation-grounded, multi-label
    synthetic_raw/*.json                   277 Claude-authored, 6 domains + Day 11
                                               finance augmentation, multi-label
    ---------------------------------------------------------------------------------
    total                                  380

DROPPED (Day 11): day7_external_promise_sample.json -- 60 external PROMISE NFR
rows. They were single-label (mean 1.00 categories/requirement) against a
ground truth averaging 2.50, so they trained the model toward the wrong output
shape, and the corpus is non-finance general software ("The website should use
Flash to provide animation on the home page"). The file is left on disk; it is
simply no longer merged. Removing it lifted the corpus mean to 2.44
categories/requirement.

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

CORRECTIONS = HERE / "label_corrections.json"
FINANCE_SPECIFIC = HERE / "day7_finance_specific.json"
REGULATION = HERE / "day7_regulation_grounded_sample.json"
SYNTHETIC_DIR = HERE / "synthetic_raw"
GROUND_TRUTH = HERE / "ground_truth.json"
OUT = HERE / "training_set.json"

MIN_PER_CATEGORY = 10  # Definition of Done


def load_reqs(path: Path) -> list[dict]:
    return json.loads(path.read_text())["requirements"]


def apply_corrections(merged: list[dict]) -> list[dict]:
    """Apply label_corrections.json on top of the merged corpus.

    Source files stay as originally authored; every deviation lives in that one
    reviewable file with a reason attached, so a correction can be audited or
    reverted on its own. Mostly this enforces the Regulatory rule recorded
    there -- see its `regulatory_rule` block.
    """
    if not CORRECTIONS.exists():
        return merged

    doc = json.loads(CORRECTIONS.read_text())
    by_id = {r["id"]: r for r in merged}
    applied = 0

    for entry in doc.get("corrections", []):
        rid = entry["id"]
        req = by_id.get(rid)
        if req is None:
            raise SystemExit(f"label_corrections.json targets unknown id {rid}")

        cats = list(req["categories"])
        for c in entry.get("remove", []):
            if c not in cats:
                raise SystemExit(
                    f"{rid}: correction removes {c!r} but it is not on the "
                    f"requirement (currently {cats}) -- the source data changed "
                    f"under the correction, re-check it"
                )
            cats.remove(c)
        for c in entry.get("add", []):
            if c not in CATEGORIES:
                raise SystemExit(f"{rid}: correction adds unknown category {c!r}")
            if c not in cats:
                cats.append(c)

        if not cats:
            raise SystemExit(f"{rid}: correction would leave no categories")
        req["categories"] = cats
        applied += 1

    print(f"\n  applied {applied} label correction(s) from {CORRECTIONS.name}")
    return merged


def main() -> None:
    sources: list[tuple[str, Path]] = [
        ("synthetic-finance", FINANCE_SPECIFIC),
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

    merged = apply_corrections(merged)

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
            "classifier. Combines finance-specific requirements, regulation-grounded "
            "synthetic requirements, and Claude-authored synthetic requirements "
            "across six domains plus the Day 11 finance augmentation. The external "
            "PROMISE-NFR rows were dropped on Day 11 as single-label and off-domain. "
            "SEPARATE from ground_truth.json, which stays held out for evaluation."
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
