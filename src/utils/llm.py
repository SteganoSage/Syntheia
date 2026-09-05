"""LLM gateway — auto-detects whether to use Gemini or OpenAI based on
which API key is set in .env.

All other files import from here:
    from src.utils.llm import call_structured

Switch providers by simply changing which key is filled in your .env file.
"""

import os

from dotenv import load_dotenv

load_dotenv()


def _has_key(name: str) -> bool:
    val = os.getenv(name, "")
    return bool(val) and not val.startswith("your_")


# Auto-detect provider and re-export call_structured
if _has_key("GEMINI_API_KEY"):
    from src.utils.llm_gemini import call_structured  # noqa: F401

    print("🟢 LLM Provider: Gemini (free)")
elif _has_key("OPENAI_API_KEY"):
    from src.utils.llm_openai import call_structured  # noqa: F401

    print("🔵 LLM Provider: OpenAI (paid)")
else:
    raise RuntimeError(
        "No LLM API key found!\n"
        "Set either GEMINI_API_KEY or OPENAI_API_KEY in your .env file.\n"
        "Gemini is free: https://aistudio.google.com/apikey"
    )
