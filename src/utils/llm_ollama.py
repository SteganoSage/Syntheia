"""Ollama LLM provider — serves both tiers of the architecture.

The core pipeline runs on a locally-served model rather than a third-party API
(architecture change, 11 Sep). There are two tiers, and **this one module drives
both**, because Ollama's HTTP API is identical on localhost and behind a tunnel:

    local tier   OLLAMA_HOST=http://localhost:11434   3-4B, offline, fallback
    cloud tier   OLLAMA_HOST=https://xxxx.ngrok.io    Llama-3.1-8B, demo

Switching tiers is one environment variable. Never fork this into two code
paths -- if the tiers diverge, the benchmark numbers stop being comparable and
the demo-day fallback stops being tested by everyday use.

    OLLAMA_HOST    base URL         (default http://localhost:11434)
    OLLAMA_MODEL   model tag        (default phi3.5)

Interface mirrors `llm_gemini.py` so `src/utils/llm.py` can route between them:
`call_structured(system_prompt, user_prompt, response_model, model, temperature)`.

Why structured output needs care here that Gemini did not:
    Gemini enforces a response schema server-side. Ollama has a `format` field
    that constrains decoding to valid JSON, but nothing forces the *shape* to
    match your model, and small models like to wrap JSON in prose or markdown
    fences. So we pass the JSON schema in the prompt, set `format`, and still
    salvage-parse the reply before giving up. phi3.5 asked for one word replies
    with "ready\\n\\n(Note: The task specifies...)" -- that is the failure this
    guards against.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, TypeVar

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

load_dotenv()

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2:3b"   # pinned Day 12 by benchmark; see evaluation/results/day12_local_llm_benchmark.json
MAX_RETRIES = 3
TIMEOUT = 300          # generous: a cold model load off NTFS can take minutes

T = TypeVar("T", bound=BaseModel)


def get_host() -> str:
    """Base URL, read at call time so tests and the runbook can switch tiers."""
    return os.getenv("OLLAMA_HOST", DEFAULT_HOST).rstrip("/")


def get_model() -> str:
    return os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)


def is_cloud() -> bool:
    """True when pointed at something other than localhost -- used by the
    pipeline to report which tier served a run."""
    host = get_host()
    return not ("localhost" in host or "127.0.0.1" in host)


# ── Health ─────────────────────────────────────────────────────────────

def is_available(timeout: float = 3.0) -> bool:
    """Is the endpoint reachable? Cheap enough to gate a fallback on."""
    try:
        r = requests.get(f"{get_host()}/api/tags", timeout=timeout)
        return r.status_code == 200
    except requests.RequestException:
        return False


def list_models() -> list[str]:
    r = requests.get(f"{get_host()}/api/tags", timeout=10)
    r.raise_for_status()
    return [m["name"] for m in r.json().get("models", [])]


# ── Raw generation ─────────────────────────────────────────────────────

def generate(
    system_prompt: str,
    user_prompt: str,
    model: str | None = None,
    temperature: float = 0.2,
    json_mode: bool = False,
    schema: dict | None = None,
) -> dict[str, Any]:
    """One /api/generate call. Returns the full response dict.

    The dict carries Ollama's own timing counters (`eval_count`,
    `eval_duration`, `load_duration`), which is why the benchmark calls this
    rather than measuring wall-clock: wall-clock on a first call is dominated
    by loading gigabytes off disk, not by generation speed.
    """
    payload: dict[str, Any] = {
        "model": model or get_model(),
        "system": system_prompt,
        "prompt": user_prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if json_mode:
        # A dict here constrains decoding to that schema; "json" only forces
        # syntactically valid JSON of any shape.
        payload["format"] = schema if schema else "json"

    r = requests.post(f"{get_host()}/api/generate", json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def call_text(
    system_prompt: str,
    user_prompt: str,
    model: str | None = None,
    temperature: float = 0.2,
) -> str:
    return generate(system_prompt, user_prompt, model, temperature)["response"]


# ── JSON salvage ───────────────────────────────────────────────────────

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)


def extract_json(text: str) -> str | None:
    """Pull the JSON object or array out of a reply that may be wrapped.

    Tried in order: the whole string, a fenced code block, then the widest
    brace/bracket span. Small models routinely emit
    "Here is the JSON:\\n```json\\n{...}\\n```\\nHope this helps!" and all three
    layers of that are recoverable.
    """
    text = text.strip()
    if not text:
        return None

    for candidate in (text, *(m.group(1) for m in _FENCE.finditer(text))):
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            span = text[start:end + 1]
            try:
                json.loads(span)
                return span
            except json.JSONDecodeError:
                continue
    return None


# ── Structured call (mirrors llm_gemini.call_structured) ───────────────

def call_structured(
    system_prompt: str,
    user_prompt: str,
    response_model: type[T],
    model: str | None = None,
    temperature: float = 0.2,
) -> T:
    """Call the model and parse the reply into `response_model`.

    Retries on malformed output, appending the validation error to the prompt
    so the model gets told what it got wrong rather than just being asked again.
    """
    schema = response_model.model_json_schema()
    schema_hint = (
        f"{system_prompt}\n\n"
        "Respond with a single JSON object and nothing else. No prose, no "
        "explanation, no markdown fences. It must validate against this schema:\n"
        f"{json.dumps(schema, indent=2)}"
    )

    prompt = user_prompt
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = generate(
                schema_hint, prompt, model, temperature,
                json_mode=True, schema=schema,
            )["response"]
        except requests.RequestException as e:
            last_error = e
            if attempt == MAX_RETRIES:
                break
            time.sleep(2 * attempt)
            continue

        payload = extract_json(raw)
        if payload is not None:
            try:
                return response_model.model_validate_json(payload)
            except ValidationError as e:
                last_error = e
                prompt = (
                    f"{user_prompt}\n\n"
                    f"Your previous reply did not match the schema:\n{e}\n"
                    "Return only the corrected JSON object."
                )
        else:
            last_error = ValueError("no JSON found in reply")
            prompt = (
                f"{user_prompt}\n\n"
                "Your previous reply contained no parseable JSON. "
                "Return only a JSON object."
            )

        if attempt < MAX_RETRIES:
            print(f"    ⚠️ {model or get_model()}: bad structured output "
                  f"(attempt {attempt}/{MAX_RETRIES}), retrying")

    raise RuntimeError(
        f"{model or get_model()} at {get_host()} failed to produce valid "
        f"{response_model.__name__} after {MAX_RETRIES} attempts. "
        f"Last error: {last_error}"
    )
