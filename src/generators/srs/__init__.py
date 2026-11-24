"""Software Requirements Specification generation (Phase C, PS §12).

    from src.generators.srs import generate_srs
    paths = generate_srs(run, "out/", layout="ieee830")

`run` is the dictionary `main.py --json` writes. The SRS is built from it, so
any saved run can be turned into a document later without re-running the
pipeline: `python -m src.generators.export run.json --out out/`.
"""


def generate_srs(*args, **kwargs):
    # Imported on call, so the deterministic helpers (derive, builder) can be
    # used and tested without python-docx or an LLM being importable.
    from src.generators.srs.generate import generate_srs as _generate

    return _generate(*args, **kwargs)
