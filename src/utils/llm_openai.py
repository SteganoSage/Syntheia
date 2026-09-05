"""OpenAI LLM provider — uses GPT-4o-mini for structured output.

Requires OPENAI_API_KEY in .env and a funded OpenAI account.
"""

import os
from typing import TypeVar

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

DEFAULT_MODEL = "gpt-4o-mini"

T = TypeVar("T", bound=BaseModel)

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_openai_api_key_here":
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add a real key to your .env file."
            )
        _client = OpenAI(api_key=api_key)
    return _client


def call_structured(
    system_prompt: str,
    user_prompt: str,
    response_model: type[T],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
) -> T:
    """Call OpenAI and parse its response directly into `response_model`."""
    client = get_client()
    completion = client.chat.completions.parse(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=response_model,
    )
    message = completion.choices[0].message
    if message.parsed is None:
        raise ValueError(f"Model did not return structured output: {message.refusal}")
    return message.parsed
