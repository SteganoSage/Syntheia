"""Check the compliance and security rule packs against the knowledge base.

    python -m notebooks.13_check_rule_packs            # resolve every source
    python -m notebooks.13_check_rule_packs --excerpt  # and print the clause text

No LLM. For every rule with a regulatory basis this resolves its `source` to
the clause the agents will cite, and prints it -- so a reviewer verifying a
pack reads the actual clause beside the rule, rather than trusting that the
query lands where its author meant it to.

Exit 1 if any regulatory rule fails to resolve: a rule that cites nothing is
claiming an authority it cannot show.
"""

from __future__ import annotations

import argparse
import textwrap

from src.agents.rulepack import load_rules, resolve_source
from src.agents.compliance import RULES_PATH
from src.agents.security import CONTROLS_PATH


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--excerpt", action="store_true", help="print each clause")
    args = ap.parse_args()

    failed = 0
    for label, path in (("compliance", RULES_PATH), ("security", CONTROLS_PATH)):
        rules = load_rules(path)
        print(f"\n== {label}: {len(rules)} rules ({path.relative_to(path.parents[2])})")
        for r in rules:
            if r.basis == "practice":
                print(f"  --    {r.id:16s} practice  {r.reference}")
                continue
            src = resolve_source(r)
            if src is None:
                failed += 1
                print(f"  FAIL  {r.id:16s} nothing in {list(r.source_titles)} "
                      f"above the floor for: {r.source_query[:70]}")
                continue
            print(f"  ok    {r.id:16s} {src.confidence:.2f}  {src.citation[:90]}")
            if r.source_min_confidence is not None:
                print(f"        (floor lowered to {r.source_min_confidence} for this "
                      f"rule -- see the comment in the pack)")
            if args.excerpt:
                body = textwrap.fill(src.excerpt[:400].replace("\n", " "), 92,
                                     initial_indent="          ",
                                     subsequent_indent="          ")
                print(body)
        unverified = sum(1 for r in rules if not r.verified)
        if unverified:
            print(f"  ({unverified}/{len(rules)} rules not yet verified by a person)")

    print(f"\n{'all regulatory rules resolve' if not failed else f'{failed} rule(s) FAILED to resolve'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
