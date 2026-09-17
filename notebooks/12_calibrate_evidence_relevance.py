"""Can a sentence embedder tell whether a citation answers the question it is attached to?

MEASURED 17 Sep 2026. ANSWER: NO. This script exists to record that, and to be
re-run if the embedding model changes.

The problem it was meant to solve: the evidence gate in `src/agents/sdlc.py`
checks that a citation is TRACEABLE to the problem statement, which catches
invented quotes and empty evidence. It cannot catch a real quote attached to the
wrong question, and after the question glosses went in that became the dominant
failure mode:

    USER-4  "Do the users want to be involved in all phases of the life cycle?"
            cited "They want quick disbursal, maybe under 10 minutes."

Both quotes are genuinely in their scenarios, so the structural gate passes them.
The idea was to reuse the RAG embedder -- already loaded, no new dependency --
and require a minimum cosine similarity between the question and the citation,
the same mechanism as MIN_CONTEXT_RELEVANCE in the retriever.

It does not separate. Twelve citations, six known-good and six known-bad, taken
verbatim from 3B and 8B runs:

    question only      real 0.081-0.620   bad 0.114-0.263
    question + gloss   real 0.071-0.597   bad 0.132-0.355

The failure is not marginal and not a tuning problem. PTR-2's correct citation
scores 0.081 -- the lowest of all twelve, below every fabrication -- because
linking "is the project a system integration project?" to "PAN details validated
against the NSDL database in real-time" requires knowing that NSDL is an external
bureau service. That is domain knowledge, not semantic proximity. Adding the
gloss (which names "regulator or bureau services" explicitly) moved it to 0.071,
i.e. not at all.

Any threshold above the fabrications drops half the real citations, including
the PTR-2 fix that moved kyc_onboarding into the right family of recommendations.

WHAT TO DO INSTEAD -- the evidence itself says the cause is batching, not
relevance scoring. Evidence is chosen once per table and pasted onto every
question in it:

    REQ-1..REQ-5   all cite ['REQ-001', 'REQ-004']
    PTR-2, PTR-3   identical reasoning sentence, different questions

One call per question is the fix that addresses the cause. 20 calls instead of 4,
measurable with notebooks/11_eval_sdlc.py --runs.

    python -m notebooks.12_calibrate_evidence_relevance
"""

from __future__ import annotations

import numpy as np

from src.rag import embedder
from src.sdlc.framework import Framework

# (question id, citation, is the citation a real answer to THAT question)
# Every row is taken verbatim from a real run -- nothing here is invented.
CASES = [
    ("USER-4", "They want quick disbursal, maybe under 10 minutes.", False),
    ("USER-4", "The entire onboarding process should take no longer than 15 minutes", False),
    ("USER-4", "then iterate based on user feedback", True),
    ("USER-2", "they've never built ML systems before", False),
    ("TEAM-2", "they've never built ML systems before", True),
    ("PTR-4", "funding is approved for 18 months", True),
    ("PTR-4", "the CEO wants to launch an MVP in 2 months", False),
    ("PTR-5", "Video KYC sessions must be recorded and stored for a minimum of 8 years", False),
    ("TEAM-4", "The product team is small -- 3 people", False),
    ("REQ-3", "Requirements will definitely change as they learn what users want", True),
    ("PTR-2", "PAN details should be validated against the NSDL database in real-time", True),
    ("PTR-6", "Testing and validation is required at every stage of the migration", True),
]


def main() -> int:
    fw = Framework.load()
    q_by_id = {q.id: q for q in fw.questions}
    citations = [c[1] for c in CASES]

    separable_anywhere = False
    for label, render in [
        ("question only", lambda q: q.text),
        ("question + gloss", lambda q: f"{q.text} {q.gloss}".strip()),
    ]:
        vq = np.array(embedder.embed([render(q_by_id[c[0]]) for c in CASES]))
        vc = np.array(embedder.embed(citations))
        sims = (vq * vc).sum(axis=1)          # embed() normalises, so dot == cosine

        good = [s for c, s in zip(CASES, sims) if c[2]]
        bad = [s for c, s in zip(CASES, sims) if not c[2]]
        separable = min(good) > max(bad)
        separable_anywhere |= separable

        print(f"\n{label}")
        print(f"  real {min(good):.3f}-{max(good):.3f}   bad {min(bad):.3f}-{max(bad):.3f}   "
              f"{'SEPARABLE' if separable else 'OVERLAP -- no threshold works'}")
        for (qid, cite, is_real), s in sorted(zip(CASES, sims), key=lambda x: -x[1]):
            note = "  <-- real, and below a fabrication" if is_real and s < max(bad) else ""
            print(f"    {s:.3f}  {qid:<7} {'real' if is_real else 'bad':<5} {cite[:46]}{note}")

    print("\nConclusion: "
          + ("a threshold exists -- revisit the decision not to ship this."
             if separable_anywhere else
             "neither framing separates. Not shipped; see the module docstring."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
