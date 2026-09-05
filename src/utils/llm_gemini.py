"""Gemini LLM provider — uses the NEW google-genai SDK with Gemini 2.0 Flash.

Requires GEMINI_API_KEY in .env.  Free tier: 15 req/min, 1M tokens/day.
Get a key at: https://aistudio.google.com/apikey
"""

import json
import os
from typing import TypeVar

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel

load_dotenv()

DEFAULT_MODEL = "gemini-3.6-flash"

T = TypeVar("T", bound=BaseModel)

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or api_key == "your_gemini_api_key_here":
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add a real key to your .env file.\n"
                "Get one for free at: https://aistudio.google.com/apikey"
            )
        _client = genai.Client(api_key=api_key)
    return _client


def call_structured(
    system_prompt: str,
    user_prompt: str,
    response_model: type[T],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
) -> T:
    """Call Gemini and parse its response directly into `response_model`."""
    client = _get_client()

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
