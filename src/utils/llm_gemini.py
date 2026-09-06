"""Gemini LLM provider — uses the google-genai SDK with Gemini 3.6 Flash.

Supports MULTIPLE API keys via round-robin rotation to bypass free-tier
rate limits (20 req/day per key). Add keys to .env as comma-separated:

    GEMINI_API_KEY=key1,key2,key3

Get keys at: https://aistudio.google.com/apikey
"""

import os
import time
from itertools import cycle
from typing import TypeVar

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from pydantic import BaseModel

load_dotenv()

DEFAULT_MODEL = "gemini-3.6-flash"
MAX_RETRIES = 5

T = TypeVar("T", bound=BaseModel)

# ── Round-robin key pool ───────────────────────────────────────────────

_clients: list[genai.Client] = []
_client_cycle = None
_key_count = 0


def _init_clients():
    global _clients, _client_cycle, _key_count
    if _clients:
        return

    raw = os.getenv("GEMINI_API_KEY", "")
    keys = [k.strip() for k in raw.split(",") if k.strip() and not k.strip().startswith("your_")]

    if not keys:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add one or more keys to .env.\n"
            "For multiple keys, comma-separate them:\n"
            "  GEMINI_API_KEY=key1,key2,key3\n"
            "Get keys for free at: https://aistudio.google.com/apikey"
        )

    _clients = [genai.Client(api_key=k) for k in keys]
    _client_cycle = cycle(range(len(_clients)))
    _key_count = len(keys)
    print(f"    🔑 Loaded {_key_count} Gemini API key(s)")


def _next_client() -> tuple[genai.Client, int]:
    """Return the next client in round-robin order."""
    _init_clients()
    idx = next(_client_cycle)
    return _clients[idx], idx


# ── Structured call with retry ─────────────────────────────────────────

def call_structured(
    system_prompt: str,
    user_prompt: str,
    response_model: type[T],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
) -> T:
    """Call Gemini and parse response into `response_model`.

    Rotates API keys round-robin. On rate limit (429/503), switches
    to the next key and retries.
    """
    _init_clients()

    for attempt in range(1, MAX_RETRIES + 1):
        client, key_idx = _next_client()
        try:
            response = client.models.generate_content(
                model=model,
                contents=user_prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=temperature,
                    response_mime_type="application/json",
                    response_schema=response_model,
                ),
            )
            parsed = response_model.model_validate_json(response.text)
            return parsed

        except (genai_errors.ClientError, genai_errors.ServerError) as e:
            error_str = str(e)
            if "429" in error_str or "503" in error_str:
                wait = 15 * attempt
                print(f"    ⚠️ Key {key_idx + 1}/{_key_count} rate limited "
                      f"(attempt {attempt}/{MAX_RETRIES}). Rotating + waiting {wait}s...")
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(f"Failed after {MAX_RETRIES} retries — all keys exhausted")
