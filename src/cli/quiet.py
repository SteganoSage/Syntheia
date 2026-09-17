"""Silence third-party console noise before anything loads a model.

Three separate libraries print to the terminal on import or first use, and all
three land in the middle of the application's own output:

    transformers        a tqdm "Loading weights" bar, once per model load --
                        twice per session (RoBERTa, then the embedder), and the
                        second one fires mid-run, right after the submit marker
    huggingface_hub     a bar for any download, plus an unauthenticated-request
                        warning when HF_TOKEN is unset
    transformers        info-level logging about weight initialisation

None of it is wrong; it is simply not addressed to the person using the app, and
it breaks the one thing a TUI has to get right, which is that the screen belongs
to the application. The one-shot CLI keeps it -- there it is useful diagnostic
output going to a log.

Import this module for its side effects, BEFORE anything imports transformers.
"""

from __future__ import annotations

import os
import warnings


def silence() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    try:
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()
        hf_logging.set_verbosity_error()
    except Exception:
        # Never let cosmetics break startup. A noisy app beats a dead one.
        pass

    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except Exception:
        pass

    warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
