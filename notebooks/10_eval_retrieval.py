"""Evaluate retrieval on its own, before it is wired into generation.

    python -m notebooks.10_eval_retrieval
    python -m notebooks.10_eval_retrieval --k 10 --show-misses

Why separately: a retrieval failure and a generation failure look identical from
the far end of the pipeline -- the requirement comes out ungrounded either way.
This project has already lost time twice to metrics that could not fail and to
diagnosing the wrong stage. So measure the retriever against known answers first,
and only then trust what the extractor does with it.

The queries below name the document that *should* answer them, chosen so the
answer lives in exactly one source. recall@k is the share of queries whose
correct document appears in the top k; MRR rewards ranking it first.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.rag import store                      # noqa: E402

REPORT = REPO / "evaluation/results/day14_retrieval_eval.json"

# (query, substring that must appear in the winning chunk's doc_path)
# Written by hand against what each document actually covers. Keep them phrased
# the way a requirements engineer would ask, not the way the source is worded --
# retrieval that only works on near-verbatim queries is not retrieval.
QUERIES: list[tuple[str, str]] = [
    ("What identity documents must be collected when opening an account?", "kyc"),
    ("How is a customer's risk category assigned and reviewed?", "kyc"),
    ("What are the requirements for video-based customer identification?", "kyc"),
    ("How long must KYC records be retained after the relationship ends?", "kyc"),
    ("When must a suspicious transaction be reported to the financial intelligence unit?", "kyc"),

    ("What controls apply when a bank outsources IT services to a third party?", "outsourcing"),
    ("What must an exit strategy for an outsourced IT arrangement cover?", "outsourcing"),
    ("What are the requirements for using cloud computing services?", "outsourcing"),

    ("Who is responsible for IT governance and what committees are required?", "it_governance"),
    ("What change management and patch management controls are required?", "it_governance"),
    ("What are the requirements for business continuity and disaster recovery?", "it_governance"),
    ("What audit trail and logging requirements apply to IT systems?", "it_governance"),

    ("What early warning signals indicate a potentially fraudulent loan account?", "fraud"),
    ("How must a fraud be classified and reported once detected?", "fraud"),

    ("What authentication is required for a card-not-present transaction?", "card"),
    ("What must be disclosed to a customer before issuing a credit card?", "card"),

    ("What consent is required before processing a person's personal data?", "dpdp"),
    ("What must a data fiduciary do when a personal data breach occurs?", "dpdp"),
    ("What rights does a data principal have over their personal data?", "dpdp"),

    ("What functional requirements apply to a loan origination workflow?", "loan_origination"),
    ("What turnaround time and alert requirements apply to loan applications?", "loan_origination"),
    ("What integration with the core banking system is required?", "loan_origination"),

    ("What requirements apply to detecting tampered or forged documents?", "fraud_detection_rfp"),
    ("What data sources should feed a fraud risk score?", "fraud_detection_rfp"),

]

# Second documents that ALSO correctly answer a query, added 14 Nov 2026 when
# the corpus grew from 13 to 19 documents. The gold set above names one source
# per query, so a new document that answers as well or better counted as a miss.
# Each addition is justified by what the document actually says. Results are
# reported BOTH ways -- strict (the original gold, unchanged) and with these --
# so an alternate can never quietly flatter the headline number.
ALSO_CORRECT: dict[str, list[str]] = {
    # PML (Maintenance of Records) Rules, rule 9: the documents to be obtained
    # for client identification -- the legal basis the KYC direction applies.
    "What identity documents must be collected when opening an account?": ["pmla"],
    # PML Rules, rules 3 and 7-8: the suspicious-transaction record and the duty
    # to furnish it. The KYC direction restates this; the Rules impose it.
    "When must a suspicious transaction be reported to the financial intelligence unit?": ["pmla"],
    # RBI Master Direction on Digital Payment Security Controls, para 33:
    # multi-factor authentication for payments through electronic modes,
    # which is the card-not-present case.
    "What authentication is required for a card-not-present transaction?": ["md7493544c24b5fc47d0ab12798c61cdb56f"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--show-misses", action="store_true",
                   help="print what was retrieved instead, for every miss")
    p.add_argument("--sector", choices=["banking", "insurance"], default=None,
                   help="filter as the pipeline does for a scenario in this sector "
                        "(every query here is written for a banking scenario)")
    p.add_argument("--out", type=Path, default=REPORT,
                   help="where to save the report (default: the Day 14 file)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    st = store.stats()
    if st["chunks"] == 0:
        print("The index is empty. Run:")
        print("  python -m notebooks.09a_fetch_sources")
        print("  python -m notebooks.09_build_knowledge_base")
        return 1

    print(f"index: {st['chunks']} chunks | embedding model: {st['embedding_model']}")
    print(f"queries: {len(QUERIES)} | k = {args.k}\n")

    from src.rag.sectors import in_sector

    sectors = {args.sector} if args.sector else None
    if sectors:
        print(f"sector filter: {args.sector}\n")
    alt_at_1 = alt_at_k = 0
    alt_rr: list[float] = []
    hits_at_1 = hits_at_k = 0
    reciprocal: list[float] = []
    top_scores: list[float] = []
    misses: list[dict] = []

    for query, expected in QUERIES:
        results = (in_sector(store.query(query, k=args.k * 3), sectors)[:args.k]
                   if sectors else store.query(query, k=args.k))
        paths = [r["metadata"].get("doc_path", "") for r in results]
        rank = next((i for i, p in enumerate(paths) if expected in p.lower()), None)
        ok = [expected] + ALSO_CORRECT.get(query, [])
        alt = next((i for i, p in enumerate(paths) if any(e in p.lower() for e in ok)), None)
        if alt == 0:
            alt_at_1 += 1
        if alt is not None:
            alt_at_k += 1
        alt_rr.append(1 / (alt + 1) if alt is not None else 0.0)

        if results:
            top_scores.append(results[0]["score"])
        if rank == 0:
            hits_at_1 += 1
        if rank is not None:
            hits_at_k += 1
            reciprocal.append(1 / (rank + 1))
            mark = "OK " if rank == 0 else f"@{rank + 1}"
        else:
            reciprocal.append(0.0)
            mark = "MISS"
            misses.append({
                "query": query, "expected": expected,
                "got": [{"doc": Path(p).name, "clause": r["metadata"].get("clause_id", ""),
                         "score": r["score"]} for p, r in zip(paths, results)][:3],
            })

        top = f"{results[0]['score']:.3f}" if results else "  -  "
        print(f"  {mark:<4} {top}  {query[:64]}")

    n = len(QUERIES)
    summary = {
        "queries": n,
        "k": args.k,
        "recall_at_1": round(hits_at_1 / n, 3),
        f"recall_at_{args.k}": round(hits_at_k / n, 3),
        "mrr": round(statistics.mean(reciprocal), 3),
        "mean_top_score": round(statistics.mean(top_scores), 3) if top_scores else None,
        "chunks": st["chunks"],
        "embedding_model": st["embedding_model"],
        "sector_filter": args.sector,
        "with_alternates": {
            "recall_at_1": round(alt_at_1 / n, 3),
            f"recall_at_{args.k}": round(alt_at_k / n, 3),
            "mrr": round(statistics.mean(alt_rr), 3),
            "alternates": len(ALSO_CORRECT),
        },
    }

    print(f"\n  recall@1   {summary['recall_at_1']:.0%}")
    print(f"  recall@{args.k}   {summary[f'recall_at_{args.k}']:.0%}")
    print(f"  MRR        {summary['mrr']:.3f}")
    print(f"  mean top score {summary['mean_top_score']}")
    w = summary["with_alternates"]
    print(f"  with {w['alternates']} justified alternates: recall@1 {w['recall_at_1']:.0%}  "
          f"recall@{args.k} {w[f'recall_at_{args.k}']:.0%}  MRR {w['mrr']:.3f}")

    if args.show_misses and misses:
        print(f"\n  {len(misses)} miss(es):")
        for m in misses:
            print(f"\n  ? {m['query']}")
            print(f"    expected a chunk from: *{m['expected']}*")
            for g in m["got"]:
                print(f"    got  {g['score']:.3f}  {g['doc']} §{g['clause']}")

    print("\n  Reading this: recall@5 below ~0.8 means grounding will be unreliable")
    print("  no matter how good the generation prompt is -- fix retrieval first.")
    print("  A high score on a miss is worse than a low one: it means the wrong")
    print("  document is confidently winning, which is what produces a citation")
    print("  that looks authoritative and is not.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "misses": misses}, indent=2) + "\n")
    print(f"\n  Saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
