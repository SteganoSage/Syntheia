"""Day 13-15: ingest the financial knowledge base into ChromaDB.

    python -m notebooks.09_build_knowledge_base            # incremental
    python -m notebooks.09_build_knowledge_base --reset    # rebuild from scratch
    python -m notebooks.09_build_knowledge_base --dry-run  # chunk, don't index

This is deliverable 1's input side: the supervisor's part 1 is "feed a
lightweight LLM existing requirement documents from a specific domain", and this
is what does the feeding.

Every document needs a sidecar manifest (see src/rag/loader.py). Ingestion fails
loudly on a document without one rather than indexing text nobody can cite.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.rag import store                                    # noqa: E402
from src.rag.chunker import chunk_document                   # noqa: E402
from src.rag.loader import MissingManifest, discover, load_document  # noqa: E402

KB_DIR = REPO / "data/knowledge_base"
REPORT = REPO / "evaluation/results/day13_knowledge_base.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--kb-dir", type=Path, default=KB_DIR)
    p.add_argument("--reset", action="store_true",
                   help="drop the collection first; use after changing the embedding model")
    p.add_argument("--dry-run", action="store_true",
                   help="load and chunk but do not embed or store")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    docs = discover(args.kb_dir)

    if not docs:
        print(f"No documents under {args.kb_dir}.")
        print("Add PDFs/DOCX/MD plus a <name>.manifest.yaml beside each one.")
        print("See docs/knowledge_base.md for the manifest fields and what qualifies.")
        return 1

    print(f"Found {len(docs)} document(s) under {args.kb_dir.relative_to(REPO)}\n")

    all_chunks, failures = [], []
    for path in docs:
        try:
            text, doc = load_document(path)
        except MissingManifest as e:
            failures.append((path.name, "no manifest"))
            print(f"  SKIP  {path.name}\n        {e}")
            continue
        except Exception as e:                    # noqa: BLE001 -- scanned PDFs land here
            failures.append((path.name, str(e)[:120]))
            print(f"  SKIP  {path.name}\n        {type(e).__name__}: {e}")
            continue

        chunks = chunk_document(text, doc)
        all_chunks.extend(chunks)
        clause_rate = sum(1 for c in chunks if c.clause_id) / len(chunks) if chunks else 0
        print(f"  OK    {path.name}")
        print(f"        {len(text):>7,} chars -> {len(chunks):>4} chunks | "
              f"{clause_rate:.0%} carry a clause id | {doc.doc_type} | {doc.jurisdiction}")
        if clause_rate < 0.5:
            # Worth saying out loud: without clause ids the citations degrade to
            # document-level, which is the difference between "§16(b)" and
            # "somewhere in this PDF".
            print(f"        ^ low clause coverage -- structure may not have survived extraction")

    if not all_chunks:
        print("\nNothing ingestable.")
        return 1

    by_type = Counter(c.doc.doc_type for c in all_chunks)
    sizes = sorted(len(c.text) for c in all_chunks)
    print(f"\n{len(all_chunks)} chunks total  |  by type: {dict(by_type)}")
    print(f"chunk chars: min {sizes[0]}  median {sizes[len(sizes)//2]}  max {sizes[-1]}")

    if args.dry_run:
        print("\n--dry-run: nothing indexed.")
        return 0

    t = time.perf_counter()
    col = store.get_collection(reset=args.reset)
    n = store.add_chunks(all_chunks, collection=col)
    print(f"\nIndexed {n} chunks in {time.perf_counter() - t:.1f}s")
    print(f"Collection now holds {col.count()} chunks ({store.stats()['embedding_model']})")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({
        "documents": len(docs),
        "ingested": len(docs) - len(failures),
        "failures": failures,
        "chunks": len(all_chunks),
        "by_doc_type": dict(by_type),
        "clause_coverage": round(
            sum(1 for c in all_chunks if c.clause_id) / len(all_chunks), 3),
        "embedding_model": store.stats()["embedding_model"],
    }, indent=2) + "\n")
    print(f"Report written to {REPORT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
