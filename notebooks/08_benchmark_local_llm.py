"""Day 12: benchmark local LLM candidates (and, later, the cloud tier).

    python -m notebooks.08_benchmark_local_llm
    python -m notebooks.08_benchmark_local_llm --models phi3.5 qwen2.5:3b
    OLLAMA_HOST=https://xxxx.ngrok.io python -m notebooks.08_benchmark_local_llm --tier cloud

Picks the local model on evidence rather than assumption, per the Day 9-11
lesson that measurement beat guesswork. Runs through `src.utils.llm_ollama`, the
same wrapper the pipeline uses, so the cloud tier is benchmarked by the same
code path and the numbers stay comparable.

What is measured, and why each one:

  tokens/sec        Ollama's own eval_count / eval_duration, NOT wall-clock.
                    A first call spends most of its time loading gigabytes off
                    NTFS -- phi3.5 measured 3 tokens in 0.10s of generation
                    after 49.26s of loading. Wall-clock would report ~500x
                    worse than reality.

  load seconds      Reported separately, because it is a real demo-day cost
                    (the first request after a restart pays it) but not a
                    property of the model's speed.

  peak VRAM         Sampled from nvidia-smi in a background thread DURING
                    generation. This is the number that decides whether a model
                    truly fits 4GB or is silently spilling to system RAM -- the
                    whole reason 8B was ruled out for local serving.

  valid JSON rate   Whether the reply parses at all, after salvage.

  in-taxonomy rate  Whether the categories it invents are actually among the 13.
                    Separate from valid-JSON on purpose: the first structured
                    call this wrapper ever made returned schema-valid output
                    with the categories "Security Requirements" and "Data
                    Protection", neither of which exists in the taxonomy.
                    Schema-valid is not taxonomy-valid.

  labels/requirement  Mean categories per requirement, and this one exists
                    because in-taxonomy rate alone was fooled. qwen2.5:3b
                    scored 100% in-taxonomy while assigning 8-10 of the 13
                    categories to EVERY requirement -- every label legal, the
                    whole output useless. Ground truth averages 2.50 and the
                    Gemini baseline 2-3, so anything above ~4 means the model
                    is spraying rather than choosing.

  requirements/run  Extraction yield, and its variance across repeats.

Nothing here scores extraction against ground_truth.json: ground truth is
32 individually-labelled requirements for the *classifier*, not an expected
extraction output for these scenarios. Comparing them would invent a metric.
Extraction quality against the Day 4 Gemini baseline is a separate step.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.agents.classification import ALL_CATEGORIES
from src.models.requirement import ExtractionResult
from src.utils import llm_ollama as O

SCENARIOS = sorted((REPO / "data/test_cases").glob("*/scenario_01.txt"))
OUT_PATH = REPO / "evaluation/results/day12_local_llm_benchmark.json"

# Sizes are informational; "fits" is decided by the measured VRAM column.
CANDIDATES = {
    "phi3.5":      {"params": "3.8B", "expected_gb": 2.2, "expected_fit": True},
    "qwen2.5:3b":  {"params": "3B",   "expected_gb": 1.9, "expected_fit": True},
    "llama3.2:3b": {"params": "3B",   "expected_gb": 2.0, "expected_fit": True},
    "qwen2.5:7b":  {"params": "7B",   "expected_gb": 4.4, "expected_fit": False},
    "llama3.1:8b": {"params": "8B",   "expected_gb": 4.9, "expected_fit": False},
}

SYSTEM_PROMPT = f"""\
You extract software requirements from problem scenarios for financial systems.

Return every distinct requirement you find. Each needs:
  - statement: one atomic "The system shall ..." sentence
  - category: one or more of EXACTLY these 13 labels, copied verbatim:
{chr(10).join(f'      {c}' for c in ALL_CATEGORIES)}

Use no category outside that list. Do not invent new ones.
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--models", nargs="*", default=None,
                   help="defaults to every installed model known to CANDIDATES")
    p.add_argument("--repeats", type=int, default=2,
                   help="runs per scenario; >1 exposes run-to-run variance")
    p.add_argument("--tier", default=None, choices=["local", "cloud"],
                   help="label for the results file (inferred from OLLAMA_HOST if omitted)")
    p.add_argument("--temperature", type=float, default=0.2)
    return p.parse_args()


# ── VRAM sampling ──────────────────────────────────────────────────────

class VramSampler:
    """Poll nvidia-smi in a thread so we catch the peak *during* generation.

    Sampling after the call would miss it: Ollama may already have released or
    resized its allocation. Meaningless against a remote endpoint, so it
    reports None there rather than the local GPU's idle figure.
    """

    def __init__(self, interval: float = 0.25):
        self.interval = interval
        self.peak_mb: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _used_mb() -> int | None:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            return int(out.stdout.strip().splitlines()[0]) if out.returncode == 0 else None
        except Exception:
            return None

    def _run(self) -> None:
        while not self._stop.is_set():
            mb = self._used_mb()
            if mb is not None and (self.peak_mb is None or mb > self.peak_mb):
                self.peak_mb = mb
            self._stop.wait(self.interval)

    def __enter__(self):
        if not O.is_cloud():
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)


# ── One run ────────────────────────────────────────────────────────────

def run_once(model: str, scenario: str, temperature: float) -> dict:
    schema = ExtractionResult.model_json_schema()
    started = time.perf_counter()

    with VramSampler() as vram:
        try:
            resp = O.generate(
                SYSTEM_PROMPT,
                f"Scenario:\n{scenario}\n\nExtract the requirements.",
                model=model, temperature=temperature,
                json_mode=True, schema=schema,
            )
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}",
                    "wall_seconds": round(time.perf_counter() - started, 2)}

    wall = time.perf_counter() - started
    raw = resp.get("response", "")

    eval_count = resp.get("eval_count") or 0
    eval_s = (resp.get("eval_duration") or 0) / 1e9
    load_s = (resp.get("load_duration") or 0) / 1e9

    payload = O.extract_json(raw)
    valid_json = payload is not None

    # Did it need salvaging, or was the raw reply already clean JSON?
    needed_salvage = False
    if valid_json:
        try:
            json.loads(raw.strip())
        except json.JSONDecodeError:
            needed_salvage = True

    n_requirements, cats, parse_error = 0, [], None
    extracted: list[dict] = []
    if valid_json:
        try:
            result = ExtractionResult.model_validate_json(payload)
            n_requirements = len(result.requirements)
            cats = [c for r in result.requirements for c in r.category]
            # Keep the actual output, not just counts: the metrics say whether a
            # model is fast and well-formed, never whether the requirements are
            # any good. Judging that needs the text.
            extracted = [{"statement": r.statement, "category": r.category}
                         for r in result.requirements]
        except Exception as e:
            parse_error = f"{type(e).__name__}: {e}"

    unknown = sorted({c for c in cats if c not in ALL_CATEGORIES})
    labels_per_req = (len(cats) / n_requirements) if n_requirements else None

    return {
        "ok": True,
        "tokens_per_sec": round(eval_count / eval_s, 2) if eval_s else None,
        "eval_tokens": eval_count,
        "eval_seconds": round(eval_s, 2),
        "load_seconds": round(load_s, 2),
        "wall_seconds": round(wall, 2),
        "peak_vram_mb": vram.peak_mb,
        "valid_json": valid_json,
        "needed_salvage": needed_salvage,
        "schema_valid": valid_json and parse_error is None,
        "schema_error": parse_error,
        "n_requirements": n_requirements,
        "n_category_labels": len(cats),
        "unknown_categories": unknown,
        "in_taxonomy_rate": round(1 - len(unknown) / len(set(cats)), 3) if cats else None,
        "labels_per_requirement": round(labels_per_req, 2) if labels_per_req else None,
        "extracted": extracted,
    }


def summarise(runs: list[dict]) -> dict:
    ok = [r for r in runs if r.get("ok")]
    if not ok:
        return {"runs": len(runs), "all_failed": True,
                "errors": [r.get("error") for r in runs][:3]}

    def avg(key):
        vals = [r[key] for r in ok if r.get(key) is not None]
        return round(statistics.mean(vals), 2) if vals else None

    reqs = [r["n_requirements"] for r in ok]
    peaks = [r["peak_vram_mb"] for r in ok if r.get("peak_vram_mb")]
    unknown = sorted({c for r in ok for c in r["unknown_categories"]})

    return {
        "runs": len(runs),
        "failed": len(runs) - len(ok),
        "tokens_per_sec": avg("tokens_per_sec"),
        "eval_seconds": avg("eval_seconds"),
        "load_seconds_first": ok[0].get("load_seconds"),
        "wall_seconds": avg("wall_seconds"),
        "peak_vram_mb": max(peaks) if peaks else None,
        "valid_json_rate": round(sum(r["valid_json"] for r in ok) / len(ok), 3),
        "salvage_rate": round(sum(r["needed_salvage"] for r in ok) / len(ok), 3),
        "schema_valid_rate": round(sum(r["schema_valid"] for r in ok) / len(ok), 3),
        "in_taxonomy_rate": avg("in_taxonomy_rate"),
        "labels_per_requirement": avg("labels_per_requirement"),
        "requirements_mean": round(statistics.mean(reqs), 1),
        "requirements_range": [min(reqs), max(reqs)],
        "unknown_categories": unknown,
    }


def main() -> None:
    args = parse_args()
    tier = args.tier or ("cloud" if O.is_cloud() else "local")

    if not O.is_available():
        raise SystemExit(
            f"No Ollama endpoint at {O.get_host()}.\n"
            "Local: `systemctl status ollama`. Cloud: check the tunnel is alive."
        )

    installed = O.list_models()
    installed_short = {m.split(":")[0] if m.endswith(":latest") else m for m in installed} | set(installed)

    if args.models:
        models = args.models
    else:
        models = [m for m in CANDIDATES if m in installed_short]

    missing = [m for m in models if m not in installed_short]
    if missing:
        raise SystemExit(f"Not installed: {missing}\nPull them, or pass --models.")
    if not models:
        raise SystemExit(f"None of {list(CANDIDATES)} are installed. Found: {installed}")

    scenarios = [(p.parent.name, p.read_text().strip()) for p in SCENARIOS]

    print(f"Endpoint   {O.get_host()}  [{tier}]")
    print(f"Models     {', '.join(models)}")
    print(f"Scenarios  {', '.join(n for n, _ in scenarios)}")
    print(f"Repeats    {args.repeats} per scenario "
          f"({len(models) * len(scenarios) * args.repeats} runs total)\n")

    results = {}
    for model in models:
        print(f"── {model} " + "─" * (60 - len(model)))
        runs = []
        for name, text in scenarios:
            for i in range(args.repeats):
                r = run_once(model, text, args.temperature)
                r["scenario"] = name
                runs.append(r)
                if r.get("ok"):
                    print(f"   {name:<18} run {i+1}  "
                          f"{r['tokens_per_sec'] or 0:>6.1f} tok/s  "
                          f"{r['n_requirements']:>2} reqs  "
                          f"json={'Y' if r['valid_json'] else 'N'}  "
                          f"vram={r['peak_vram_mb'] or '-'}MB"
                          + (f"  unknown={r['unknown_categories']}" if r["unknown_categories"] else ""))
                else:
                    print(f"   {name:<18} run {i+1}  FAILED: {r['error'][:60]}")
        results[model] = {"meta": CANDIDATES.get(model, {}), "summary": summarise(runs), "runs": runs}
        print()

    # ── Table ──────────────────────────────────────────────────────────
    print("=" * 88)
    print(f"  BENCHMARK — {tier} tier")
    print("=" * 88)
    print(f"  {'model':<14}{'tok/s':>8}{'VRAM MB':>9}{'JSON':>7}"
          f"{'taxon':>7}{'lbl/req':>9}{'reqs':>7}")
    print(f"  {'-' * 82}")
    for model, r in results.items():
        s = r["summary"]
        if s.get("all_failed"):
            print(f"  {model:<14}{'ALL RUNS FAILED':>40}")
            continue
        lpr = s.get('labels_per_requirement') or 0
        flag = "  <-- spraying" if lpr > 4 else ""
        print(f"  {model:<14}{s['tokens_per_sec'] or 0:>8.1f}"
              f"{s['peak_vram_mb'] or 0:>9}{s['valid_json_rate']:>7.0%}"
              f"{(s['in_taxonomy_rate'] or 0):>7.0%}{lpr:>9.1f}"
              f"{s['requirements_mean']:>7.1f}{flag}")

    for model, r in results.items():
        unk = r["summary"].get("unknown_categories")
        if unk:
            print(f"\n  {model} invented categories outside the taxonomy: {unk}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tier": tier,
        "endpoint": O.get_host(),
        "temperature": args.temperature,
        "repeats_per_scenario": args.repeats,
        "scenarios": [n for n, _ in scenarios],
        "taxonomy": ALL_CATEGORIES,
        "results": results,
    }
    existing = json.loads(OUT_PATH.read_text()) if OUT_PATH.exists() else {}
    if not isinstance(existing, dict) or "tiers" not in existing:
        existing = {"tiers": {}}
    existing["tiers"][tier] = payload          # keep local and cloud side by side
    OUT_PATH.write_text(json.dumps(existing, indent=2) + "\n")
    print(f"\n  Saved to {OUT_PATH.relative_to(REPO)} (tier: {tier})")

    print("\n  Reading this table: tok/s is throughput, but it is the last thing "
          "to decide on.\n  lbl/req above ~4 means the model is spraying "
          "categories rather than choosing\n  (ground truth averages 2.5). And "
          "no metric here reads the statements --\n  print them and judge "
          "atomicity, grounding and phrasing before pinning a winner.")


if __name__ == "__main__":
    main()
