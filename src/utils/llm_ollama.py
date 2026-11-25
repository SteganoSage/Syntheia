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
    OLLAMA_MODEL   model tag        (default llama3.2:3b)
    OLLAMA_THINK   0 / 1            reasoning models only; unset sends nothing

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
from collections.abc import Callable
from typing import Any, TypeVar

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

load_dotenv()

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2:3b"   # pinned Day 12 by benchmark; see evaluation/results/day12_local_llm_benchmark.json
MAX_RETRIES = 3
TIMEOUT = 300          # generous: a cold model load off NTFS can take minutes

# Context length trades directly against speed on a 4GB GPU, because the KV
# cache scales with it and this card has no room to spare. Measured on
# llama3.2:3b:
#
#   num_ctx 4096   model occupies ~2.3 GB, fully GPU-resident, fast
#   num_ctx 8192   model occupies ~3.08 GB, 0.78 GB spills to system RAM,
#                  and the KYC scenario did not finish extraction in 6.5 min
#                  (it completed comfortably at 4096)
#
# So 4096 stays. The way to buy room is to send less, not to raise this:
# dropping the duplicated schema from the system prompt (see call_structured)
# freed ~605 tokens per call for nothing, and batching the quality agent
# per-requirement would free far more.
#
# Raise it only when serving from the cloud tier, where VRAM is not the
# constraint -- set OLLAMA_NUM_CTX in that environment.
DEFAULT_NUM_CTX = 4096
DEFAULT_KEEP_ALIVE = "-1"   # keep the model resident; see get_keep_alive()

# Hard ceiling on generated tokens. Ollama's default is unbounded, and when the
# context fills it SHIFTS the window and keeps going -- so a model that falls
# into a repetition loop generates forever rather than hitting a wall. Two runs
# were lost to this: a single extraction call ran 11 minutes with the socket
# open and the GPU busy, which from outside is indistinguishable from a hang.
#
# The largest legitimate output measured here is ~1150 tokens (extraction, 7
# requirements), so 2048 leaves generous headroom while turning an infinite loop
# into a truncated reply that fails validation and retries visibly.
DEFAULT_NUM_PREDICT = 2048

T = TypeVar("T", bound=BaseModel)


def get_host() -> str:
    """Base URL, read at call time so tests and the runbook can switch tiers."""
    return os.getenv("OLLAMA_HOST", DEFAULT_HOST).rstrip("/")


def get_model() -> str:
    return os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)


def get_num_ctx() -> int:
    return int(os.getenv("OLLAMA_NUM_CTX", DEFAULT_NUM_CTX))


def get_num_predict() -> int:
    return int(os.getenv("OLLAMA_NUM_PREDICT", DEFAULT_NUM_PREDICT))


def get_think() -> bool | None:
    """Whether a reasoning model (qwen3, qwen3.5) thinks before answering.

    Unset means "send nothing", so models without the capability -- llama3.1,
    llama3.2 -- get exactly the request they always did. A thinking model left
    to its default spends thousands of tokens per call reasoning before the
    JSON, across a pipeline that makes dozens of calls; OLLAMA_THINK=0 is how
    such a model is compared like for like.
    """
    raw = os.getenv("OLLAMA_THINK", "").strip().lower()
    if not raw:
        return None
    return raw not in {"0", "false", "no", "off"}


def get_keep_alive() -> int | str:
    """How long Ollama holds the model in VRAM after a call.

    Ollama's default is 5 minutes, which is shorter than a pipeline run: on the
    KYC scenario the classification node took 406 s, so the LLM was evicted
    *during* classification and paid a second 71.5 s reload for the quality
    stage. Weights live on /mnt/f -- v9fs, WSL's plan9 bridge -- where that
    reload is expensive.

    "-1" pins the model until something else needs the VRAM. Sent per request
    rather than set in the systemd unit so the setting travels with the code
    and applies to the cloud tier too.
    """
    raw = os.getenv("OLLAMA_KEEP_ALIVE", DEFAULT_KEEP_ALIVE)
    # Ollama accepts an integer number of seconds (-1 meaning "never unload")
    # or a duration string such as "30m". It rejects the *string* "-1" with a
    # 400, so a numeric setting has to be sent as a JSON number.
    try:
        return int(raw)
    except (TypeError, ValueError):
        return raw


def is_cloud() -> bool:
    """True when pointed at something other than localhost -- used by the
    pipeline to report which tier served a run."""
    host = get_host()
    return not ("localhost" in host or "127.0.0.1" in host)


# ── Health ─────────────────────────────────────────────────────────────

def is_available(timeout: float | None = None) -> bool:
    """Is the endpoint reachable? Cheap enough to gate a fallback on.

    The timeout scales with distance. 3 s is generous for localhost and too
    tight for a Cloudflare tunnel, which has to complete a TLS handshake and
    route through an edge PoP before Ollama sees the request at all -- a cold
    tunnel answering /api/tags in 4 s would be judged dead, and the gateway
    would silently pick a different provider.
    """
    if timeout is None:
        timeout = 15.0 if is_cloud() else 3.0
    try:
        r = requests.get(f"{get_host()}/api/tags", timeout=timeout)
        return r.status_code == 200
    except requests.RequestException:
        return False


def load_model(model: str | None = None, timeout: float = 600.0) -> bool:
    """Pull the model into VRAM without generating anything.

    Ollama treats a request with no `prompt` as a load instruction, so this
    pays the load cost at a moment of our choosing rather than inside the
    user's first request. On this laptop that is ~78 s off /mnt/f; on the
    cloud tier it is the ~70 s of silence that most likely caused the
    Cloudflare 524 on the KYC scenario, which is why the demo runbook warms
    the endpoint before the audience arrives.
    """
    try:
        r = requests.post(
            f"{get_host()}/api/generate",
            json={"model": model or get_model(), "keep_alive": get_keep_alive()},
            timeout=timeout,
        )
        r.raise_for_status()
        return True
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
    on_token: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """One /api/generate call. Returns the full response dict.

    The dict carries Ollama's own timing counters (`eval_count`,
    `eval_duration`, `load_duration`), which is why the benchmark calls this
    rather than measuring wall-clock: wall-clock on a first call is dominated
    by loading gigabytes off disk, not by generation speed.

    **Streams by default**, reassembling the pieces so the returned dict is
    shaped exactly like a non-streamed reply -- callers see no difference.
    Streaming is not faster; it changes two things that matter anyway:

      - A blocking request sends nothing for minutes and then everything at
        once. Cloudflare's quick tunnel reads that as a dead origin and
        returns 524: the cloud tier failed the KYC scenario on both runs for
        exactly this reason, while the two shorter scenarios squeaked under
        the limit. A streamed reply keeps bytes moving, so the timeout never
        arms.
      - `timeout` stops meaning "total wall-clock" and starts meaning "gap
        between chunks", which is the thing actually worth giving up on.

    `on_token` receives each piece as it arrives -- the hook a progress
    display or a UI would use. Set OLLAMA_STREAM=0 to force the old blocking
    behaviour for an A/B.
    """
    stream = os.getenv("OLLAMA_STREAM", "1") != "0"
    payload: dict[str, Any] = {
        "model": model or get_model(),
        "system": system_prompt,
        "prompt": user_prompt,
        "stream": stream,
        "keep_alive": get_keep_alive(),
        "options": {
            "temperature": temperature,
            "num_ctx": get_num_ctx(),
            "num_predict": get_num_predict(),
        },
    }
    if json_mode:
        # A dict here constrains decoding to that schema; "json" only forces
        # syntactically valid JSON of any shape.
        payload["format"] = schema if schema else "json"
    think = get_think()
    if think is not None:
        payload["think"] = think

    url = f"{get_host()}/api/generate"

    if not stream:
        r = requests.post(url, json=payload, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()

    pieces: list[str] = []
    final: dict[str, Any] = {}
    with requests.post(url, json=payload, timeout=TIMEOUT, stream=True) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue          # keep-alive or partial frame; not fatal
            piece = chunk.get("response", "")
            if piece:
                pieces.append(piece)
                if on_token is not None:
                    on_token(piece)
            if chunk.get("done"):
                final = chunk     # the last frame carries every counter

    # Rebuild the non-streamed shape. The final frame's own "response" is the
    # empty tail, so it must be overwritten rather than trusted.
    final["response"] = "".join(pieces)
    return final


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

    # The schema is passed to Ollama's `format` field below, which constrains
    # decoding to it. Repeating it as prose in the system prompt costs ~595
    # tokens per call of a 4096-token budget (measured on QualityReport) and
    # buys nothing, so it is deliberately NOT included here. Keep it that way:
    # the instruction is what the model needs, the grammar does the enforcing.
    schema_hint = (
        f"{system_prompt}\n\n"
        "Respond with a single JSON object and nothing else. "
        "No prose, no explanation, no markdown fences."
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
