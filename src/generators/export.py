"""Turn a saved pipeline run into requirement artefacts.

    python -m src.generators.export RUN.json --out out/            # IEEE 830 SRS
    python -m src.generators.export RUN.json --out out/ --layout iso29148
    python -m src.generators.export RUN.json --out out/ --no-llm   # fully offline

RUN.json is what `main.py --json` writes. Exporting from a saved run means the
documents can be regenerated -- after a template change, or in the other
layout -- without re-running the pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="syntheia-export", description=__doc__.splitlines()[0])
    ap.add_argument("run", type=Path, help="a run JSON written by main.py --json")
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--layout", choices=["ieee830", "iso29148", "both"], default="ieee830")
    ap.add_argument("--no-llm", action="store_true",
                    help="deterministic narrative and stakeholder-grouped features; no model call")
    args = ap.parse_args(argv)

    if not args.run.exists():
        print(f"export: no such run file: {args.run}", file=sys.stderr)
        return 2
    run = json.loads(args.run.read_text(encoding="utf-8"))
    if not run.get("requirements"):
        print("export: the run has no requirements to specify", file=sys.stderr)
        return 2

    from src.generators.srs import generate_srs
    from src.utils.similarity import default_similarity

    generate_srs(run, args.out, layout=args.layout, use_llm=not args.no_llm,
                 sim=default_similarity())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
