"""Load everything the pipeline needs *before* the first real request.

Day 13. The KYC scenario took 611 s end to end, and 96% of that was loading
rather than computing:

    import transformers   171 s     venv on /mnt/f (v9fs)
    import torch           54 s     same
    RoBERTa weights       128 s     -> 1.6 s once moved to ext4
    Ollama model load      78 s     x2, because the 5-minute keep-alive
                                    expired *during* classification
    actual inference      0.2 s     30 ms/requirement, as documented

None of that is per-request work -- it is paid once per process. A script that
runs the pipeline and exits pays all of it every time; a process that stays up
pays it once. So the fix is not to make loading faster (though ext4 and
keep_alive did that), it is to move loading off the user's critical path.

Call `warm()` at startup, before accepting input. The first request then costs
what it should.
"""

from __future__ import annotations

import time

_WARMED = False


def warm(verbose: bool = True) -> dict[str, float]:
    """Load the classifier and pin the LLM in VRAM. Returns per-stage seconds.

    Safe to call more than once -- the classifier is cached by
    `get_classifier`, and the Ollama load is idempotent. Never raises: a warm
    that fails should degrade to a slow first request, not a dead process.
    """
    global _WARMED
    timings: dict[str, float] = {}

    if verbose:
        print("Warming up:")

    # ── Classifier: imports transformers, then reads ~477 MB of weights ──
    t0 = time.perf_counter()
    try:
        from src.agents.classification import get_classifier, resolve_model_dir

        model_dir = resolve_model_dir()
        if model_dir is None:
            if verbose:
                print("  classifier   no trained model found -- will use the LLM baseline")
        else:
            get_classifier()
            timings["classifier"] = time.perf_counter() - t0
            if verbose:
                print(f"  classifier   {timings['classifier']:6.1f}s  {model_dir.name}")
    except Exception as e:                      # noqa: BLE001 -- warming is best-effort
        if verbose:
            print(f"  classifier   FAILED ({type(e).__name__}: {e})")

    # ── Embedder: the knowledge-base encoder. Loaded inside the first retrieval
    # otherwise, which put ~10 s of model loading into the timed pipeline on
    # every one-shot run (14 Nov breakdown: retrieval 13 s, of it ~1 s search).
    t0 = time.perf_counter()
    try:
        from src.rag.embedder import get_model, get_model_name

        get_model()
        timings["embedder"] = time.perf_counter() - t0
        if verbose:
            print(f"  embedder     {timings['embedder']:6.1f}s  "
                  f"{get_model_name().rsplit('/', 1)[-1]}")
    except Exception as e:                      # noqa: BLE001 -- warming is best-effort
        if verbose:
            print(f"  embedder     FAILED ({type(e).__name__}: {e})")

    # ── LLM: an empty prompt loads the model without generating anything ──
    t0 = time.perf_counter()
    try:
        from src.utils import llm_ollama as O

        if O.is_available():
            loaded = O.load_model()
            tier = "cloud" if O.is_cloud() else "local"
            if loaded:
                timings["llm"] = time.perf_counter() - t0
                if verbose:
                    print(f"  llm          {timings['llm']:6.1f}s  {O.get_model()} [{tier}]")
            elif verbose:
                # Reachable but refused the load -- a bad model tag or a
                # malformed option. Say so: a warm-up that reports success on
                # failure is worse than none, because the real error then
                # surfaces later and somewhere else.
                # Name the actual mismatch rather than a list of things to
                # check. The overwhelmingly common cause is setting OLLAMA_HOST
                # to a tunnel and forgetting OLLAMA_MODEL, so the default 3B is
                # requested from a server that only has the 8B -- and "check
                # OLLAMA_MODEL" does not say that.
                print(f"  llm          LOAD REFUSED  {O.get_model()} [{tier}]")
                available = _models_on(O)
                if available and O.get_model() not in available:
                    print(f"               {O.get_model()!r} is not on that "
                          f"server. It has: {', '.join(available)}")
                    print(f"               export OLLAMA_MODEL={available[0]}")
                else:
                    print(f"               check OLLAMA_MODEL and "
                          f"OLLAMA_KEEP_ALIVE")
        elif verbose:
            print(f"  llm          unreachable at {O.get_host()}")
    except Exception as e:                      # noqa: BLE001
        if verbose:
            print(f"  llm          FAILED ({type(e).__name__}: {e})")

    _WARMED = True
    if verbose and timings:
        print(f"  total        {sum(timings.values()):6.1f}s\n")
    return timings


def _models_on(O) -> list[str]:
    """What the endpoint actually serves. Empty list if it cannot be asked."""
    try:
        import requests

        r = requests.get(f"{O.get_host()}/api/tags", timeout=10)
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def is_warmed() -> bool:
    return _WARMED
