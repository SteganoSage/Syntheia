"""Run JSON in, SRS files out."""

from __future__ import annotations

from pathlib import Path

from src.generators.srs import builder, derive
from src.generators.srs.analysis_models import write_data_model, write_interfaces, write_use_cases
from src.generators.srs.docx_render import render_docx
from src.generators.srs.md_render import render_markdown
from src.generators.srs.narrative import fallback_narrative, write_narrative


# What --srs-format / --format accept. Markdown is the default: it is what gets
# read while the content is still changing, and Word is only worth opening
# once it settles.
FORMATS = {"md": ("md",), "docx": ("docx",), "both": ("docx", "md")}


def generate_srs(run: dict, out_dir, layout: str = "ieee830", use_llm: bool = True,
                 formats=("md",), sim=None, verbose: bool = True) -> dict[str, Path]:
    """Write the SRS for a run. Returns {format: path}.

    `layout` is "ieee830", "iso29148" or "both". `formats` holds "md" and/or
    "docx".

    `use_llm` controls only the prose: narrative, use-case flows, data model and
    interface table. Everything else is deterministic. With it off the document
    is complete but plainer -- which is what an offline export of a saved run
    should be.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reqs = run.get("requirements") or []
    scenario = run.get("scenario_text", "")

    if sim is None:
        from src.utils.similarity import default_similarity

        sim = default_similarity()

    narrative = (write_narrative(scenario, reqs) if use_llm
                 else fallback_narrative(scenario, reqs))
    functional = [r for r in reqs
                  if derive.primary_section(r["statement"], r.get("category") or []) == "functional"]
    use_cases, how = write_use_cases(functional, scenario, reqs, use_llm=use_llm)
    data_model = write_data_model(scenario, reqs, use_llm=use_llm)
    interfaces = write_interfaces(scenario, reqs, use_llm=use_llm)

    # All of the above is layout-independent: one LLM pass serves both layouts.
    layouts = ["ieee830", "iso29148"] if layout == "both" else [layout]
    paths: dict[str, Path] = {}
    for lay in layouts:
        doc = builder.build(run, narrative, use_cases, sim, layout=lay,
                            data_model=data_model, interfaces=interfaces)
        stem = "srs" if lay == "ieee830" else "srs_iso29148"
        written = []
        if "docx" in formats:
            key = "docx" if len(layouts) == 1 else f"{lay}.docx"
            paths[key] = out_dir / f"{stem}.docx"
            render_docx(doc, paths[key])
            written.append(paths[key])
        if "md" in formats:
            key = "md" if len(layouts) == 1 else f"{lay}.md"
            paths[key] = out_dir / f"{stem}.md"
            paths[key].write_text(render_markdown(doc), encoding="utf-8")
            written.append(paths[key])
        if verbose:
            blocks = [b for _, s in doc.walk() for b in s.blocks if type(b).__name__ == "Requirement"]
            kinds = {}
            for b in blocks:
                t = dict(b.attributes)["Type"].split(" — ")[0]
                kinds[t] = kinds.get(t, 0) + 1
            print(f"SRS ({doc.standard.split(',')[0]}): {len(blocks)} requirements "
                  f"({', '.join(f'{v} {k.lower()}' for k, v in kinds.items())}), "
                  f"{len(use_cases)} use cases [{how}], {len(data_model.entities)} entities, "
                  f"{len(interfaces.rows)} interfaces, narrative [{narrative.source}"
                  + (f", {narrative.dropped} unquoted item(s) dropped" if narrative.dropped else "")
                  + "] -> " + ", ".join(str(p) for p in written))
    return paths
