"""LLM gateway — routes to Ollama (local or cloud) or the Gemini/OpenAI API.

All other modules import from here:

    from src.utils.llm import call_structured

Selection order, first match wins:

    1. LLM_PROVIDER=ollama|gemini|openai   explicit, always respected
    2. an Ollama endpoint is reachable     the default the architecture wants
    3. OLLAMA_HOST set but unreachable     RAISES -- never falls through
    4. GEMINI_API_KEY is set               baseline / data-generation fallback
    5. OPENAI_API_KEY is set

Step 3 is the one that had to be added. With a dead tunnel the gateway used to
drop silently to Gemini, so a run that was supposed to exercise the cloud tier
quietly became a third-party API call -- visible only as one line of banner
text among the normal startup output.

Ollama leads because of the 11 Sep architecture change: the shipping pipeline
runs on our own model, not a third-party API. Gemini stays reachable as the
baseline every fine-tuned model is measured against, and as a data-generation
tool -- but it is no longer what you get by accident.

Which tier Ollama serves is a separate question answered by OLLAMA_HOST inside
`llm_ollama.py`: localhost for the local model, a tunnel URL for the cloud one.
The gateway does not care which.

To force the baseline for a comparison run:

    LLM_PROVIDER=gemini python -m src.graph
"""

import os

from dotenv import load_dotenv

load_dotenv()


def _has_key(name: str) -> bool:
    val = os.getenv(name, "")
    return bool(val) and not val.startswith("your_")


def _resolve() -> str:
    explicit = os.getenv("LLM_PROVIDER", "").strip().lower()
    if explicit:
        if explicit not in ("ollama", "gemini", "openai"):
            raise RuntimeError(
                f"LLM_PROVIDER={explicit!r} is not one of: ollama, gemini, openai"
            )
        return explicit

    # Probe Ollama rather than assuming: a missing local daemon should fall back
    # quietly to the baseline instead of breaking every import.
    #
    # But NOT when OLLAMA_HOST is set. Naming an endpoint is an instruction, and
    # quietly routing to a different vendor because it did not answer is the
    # worst possible response: it violates "no third-party API in the core path"
    # invisibly, and the only evidence is one line of banner text that reads as
    # normal. A dead tunnel must look like a dead tunnel.
    asked_for_ollama = bool(os.getenv("OLLAMA_HOST", "").strip())
    try:
        from src.utils.llm_ollama import get_host, is_available

        if is_available():
            return "ollama"
        if asked_for_ollama:
            raise RuntimeError(
                f"OLLAMA_HOST is set to {get_host()} but nothing answered there.\n"
                f"  - is the Colab/Kaggle notebook still running?\n"
                f"  - quick-tunnel URLs change on every restart -- re-copy it\n"
                f"  - check it by hand:  curl -s {get_host()}/api/tags\n"
                f"  - to drop back to the local model:  unset OLLAMA_HOST\n"
                f"  - to use the baseline on purpose:   LLM_PROVIDER=gemini\n"
                f"Refusing to fall back silently: Gemini is a baseline, never "
                f"the shipping path."
            )
    except RuntimeError:
        raise
    except Exception:
        if asked_for_ollama:
            raise

    if _has_key("GEMINI_API_KEY"):
        return "gemini"
    if _has_key("OPENAI_API_KEY"):
        return "openai"

    raise RuntimeError(
        "No LLM provider available.\n"
        "  - start Ollama:  sudo systemctl start ollama\n"
        "  - or point at the cloud tier:  OLLAMA_HOST=https://xxxx.ngrok.io\n"
        "  - or set GEMINI_API_KEY in .env (baseline only)"
    )


PROVIDER = _resolve()

if PROVIDER == "ollama":
    from src.utils.llm_ollama import call_structured, get_host, get_model, is_cloud  # noqa: F401

    _tier = "cloud" if is_cloud() else "local"
    print(f"🦙 LLM Provider: Ollama [{_tier}] {get_model()} @ {get_host()}")
elif PROVIDER == "gemini":
    from src.utils.llm_gemini import call_structured  # noqa: F401

    print("🟢 LLM Provider: Gemini (baseline — not the shipping path)")
else:
    from src.utils.llm_openai import call_structured  # noqa: F401

    print("🔵 LLM Provider: OpenAI (baseline — not the shipping path)")
