"""Syntheia — requirements analysis UI.

    streamlit run app/streamlit_app.py

Shows the Phase 1 pipeline working end to end:

    scenario -> Extraction (local LLM) -> Classification (fine-tuned RoBERTa)
             -> Quality (local LLM) -> structured requirements

Two things this is built around:

1. **Per-stage progress, not one spinner.** The pipeline takes minutes on a
   4GB GPU -- 7 minutes on the larger scenarios. A single spinner for that long
   is unwatchable and tells an audience nothing. Each stage reports as it
   starts and finishes, with its own timing, so the architecture is visible
   while it runs.

2. **The tier is always on screen.** Local vs cloud, which model, which
   classifier. On demo day you should never have to guess whether the tunnel
   was live -- and the fallback story is only convincing if the audience can
   see it switch.

The agents are called directly rather than through the compiled LangGraph, so
each node can be timed and rendered as it completes. src/graph.py remains the
canonical pipeline definition; this mirrors its order.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

st.set_page_config(page_title="Syntheia", page_icon="📋", layout="wide")

CATEGORY_COLOURS = {
    "Business": "#2563eb", "Stakeholder": "#7c3aed", "Functional": "#0891b2",
    "Security": "#dc2626", "Privacy": "#db2777", "Regulatory": "#ea580c",
    "Performance": "#ca8a04", "Availability": "#16a34a", "Usability": "#0d9488",
    "Data-management": "#4f46e5", "Integration": "#9333ea",
    "Audit/Reporting": "#c026d3", "Operational": "#57534e",
}
SEVERITY_ICON = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🔵"}


@st.cache_resource(show_spinner=False)
def get_runtime():
    """Import the agents once. Loading RoBERTa costs ~40s, so never per-run."""
    from src.agents.classification import resolve_model_dir
    from src.utils import llm_ollama as O
    return {"resolve_model_dir": resolve_model_dir, "ollama": O}


def sidebar() -> dict:
    rt = get_runtime()
    O = rt["ollama"]

    st.sidebar.title("📋 Syntheia")
    st.sidebar.caption("Requirements analysis for financial systems")

    st.sidebar.subheader("Pipeline")
    reachable = O.is_available()
    tier = "cloud" if O.is_cloud() else "local"
    st.sidebar.markdown(
        f"{'🟢' if reachable else '🔴'} **LLM** — `{O.get_model()}` "
        f"({'☁️ cloud' if tier == 'cloud' else '💻 local'})"
    )
    st.sidebar.caption(f"`{O.get_host()}`")
    if not reachable:
        st.sidebar.error("No LLM endpoint. Start Ollama, or set OLLAMA_HOST.")

    model_dir = rt["resolve_model_dir"]()
    if model_dir:
        st.sidebar.markdown(f"🟢 **Classifier** — `{model_dir.name}`")
        st.sidebar.caption("fine-tuned, 80.2% micro-F1 on held-out ground truth")
    else:
        st.sidebar.markdown("🟡 **Classifier** — none trained, using LLM fallback")

    st.sidebar.divider()
    st.sidebar.subheader("Scenario")
    cases = sorted((REPO / "data/test_cases").glob("*/scenario_01.txt"))
    labels = ["(write my own)"] + [p.parent.name.replace("_", " ").title() for p in cases]
    choice = st.sidebar.radio("Load an example", labels, label_visibility="collapsed")

    preset = ""
    if choice != "(write my own)":
        preset = cases[labels.index(choice) - 1].read_text().strip()

    st.sidebar.divider()
    st.sidebar.caption(
        "Extraction and quality run on a locally-served model. "
        "Classification uses a BERT-family model fine-tuned on 445 labelled "
        "requirements. No third-party API is called."
    )
    return {"preset": preset, "reachable": reachable}


def render_requirement(req, idx: int) -> None:
    with st.container(border=True):
        left, right = st.columns([5, 2])
        with left:
            st.markdown(f"**{req.id}** &nbsp; {req.statement}")
            if req.category:
                chips = " ".join(
                    f"<span style='background:{CATEGORY_COLOURS.get(c, '#64748b')};"
                    f"color:white;padding:2px 9px;border-radius:11px;"
                    f"font-size:0.72rem;margin-right:4px;white-space:nowrap'>{c}</span>"
                    for c in req.category
                )
                st.markdown(chips, unsafe_allow_html=True)
            else:
                st.caption("no categories assigned")
        with right:
            st.markdown(
                f"<div style='text-align:right'>"
                f"<small>priority</small><br><b>{req.priority}</b><br>"
                f"<small>risk</small><br><b>{req.risk_level}</b></div>",
                unsafe_allow_html=True,
            )
        detail = {
            "Stakeholder": req.source_stakeholder,
            "Justification": req.business_justification,
            "Acceptance criteria": req.acceptance_criteria,
            "Assumptions": req.assumptions,
            "Regulations": ", ".join(req.applicable_regulations) if req.applicable_regulations else "",
            "Depends on": ", ".join(req.dependencies) if req.dependencies else "",
        }
        detail = {k: v for k, v in detail.items() if v}
        if detail:
            with st.expander("details"):
                for k, v in detail.items():
                    st.markdown(f"**{k}** — {v}")


def render_quality(report, requirements) -> None:
    if report is None or not report.result_list:
        st.info("No quality report produced.")
        return

    by_id = {r.requirement_id: r for r in report.result_list}
    passed = sum(1 for r in report.result_list if r.passed)
    flagged = len(report.result_list) - passed

    c1, c2, c3 = st.columns(3)
    c1.metric("Analysed", len(report.result_list))
    c2.metric("Passed", passed)
    c3.metric("Flagged", flagged)

    for req in requirements:
        result = by_id.get(req.id)
        if result is None or not result.issues:
            continue
        with st.container(border=True):
            st.markdown(f"**{req.id}** &nbsp; {req.statement}")
            for issue in result.issues:
                st.markdown(
                    f"{SEVERITY_ICON.get(issue.severity, '⚪')} "
                    f"**{issue.issue_type}** ({issue.severity}) — {issue.description}"
                )
                if issue.suggestion:
                    st.caption(f"→ {issue.suggestion}")


def run_pipeline(text: str) -> dict:
    """Run the three stages, rendering progress as each completes."""
    from src.agents.classification import classify_requirements
    from src.agents.extraction import extract_requirements
    from src.agents.quality import analyse_requirements

    out: dict = {"requirements": [], "report": None, "timings": {}}

    with st.status("Running pipeline…", expanded=True) as status:
        t = time.perf_counter()
        st.write("**1/3 Extraction** — local LLM reading the scenario…")
        result = extract_requirements(text)
        out["requirements"] = result.requirements
        out["timings"]["extraction"] = time.perf_counter() - t
        st.write(f"✓ {len(result.requirements)} requirements "
                 f"({out['timings']['extraction']:.1f}s)")

        t = time.perf_counter()
        st.write("**2/3 Classification** — fine-tuned RoBERTa assigning categories…")
        out["requirements"] = classify_requirements(out["requirements"], method="auto")
        out["timings"]["classification"] = time.perf_counter() - t
        n_labels = sum(len(r.category) for r in out["requirements"])
        st.write(f"✓ {n_labels} category labels "
                 f"({out['timings']['classification']:.1f}s)")

        t = time.perf_counter()
        st.write("**3/3 Quality** — local LLM checking for issues…")
        try:
            out["report"] = analyse_requirements(out["requirements"])
            out["timings"]["quality"] = time.perf_counter() - t
            st.write(f"✓ analysed ({out['timings']['quality']:.1f}s)")
        except Exception as e:
            out["timings"]["quality"] = time.perf_counter() - t
            st.write(f"⚠️ quality stage failed: {type(e).__name__}")
            out["quality_error"] = str(e)

        status.update(label=f"Done in {sum(out['timings'].values()):.1f}s",
                      state="complete", expanded=False)
    return out


def main() -> None:
    cfg = sidebar()

    st.title("Requirements Analysis")
    st.caption("Describe a system in plain language. The pipeline extracts "
               "requirements, classifies them, and flags quality issues.")

    text = st.text_area(
        "Problem scenario",
        value=cfg["preset"],
        height=170,
        placeholder="e.g. We need an online loan application portal where customers "
                    "can apply from their mobile phones…",
    )

    go = st.button("Analyse", type="primary", disabled=not cfg["reachable"])

    if go and text.strip():
        st.session_state["result"] = run_pipeline(text.strip())
    elif go:
        st.warning("Enter a scenario first.")

    result = st.session_state.get("result")
    if not result:
        return

    reqs = result["requirements"]
    timings = result["timings"]

    st.divider()
    m = st.columns(4)
    m[0].metric("Requirements", len(reqs))
    m[1].metric("Category labels", sum(len(r.category) for r in reqs))
    m[2].metric("Total time", f"{sum(timings.values()):.1f}s")
    m[3].metric("Slowest stage", max(timings, key=timings.get) if timings else "—")

    tab_reqs, tab_quality, tab_timing = st.tabs(
        ["Requirements", "Quality report", "Stage timing"]
    )

    with tab_reqs:
        for i, req in enumerate(reqs):
            render_requirement(req, i)

    with tab_quality:
        if result.get("quality_error"):
            st.error(f"Quality analysis failed: {result['quality_error']}")
        else:
            render_quality(result["report"], reqs)

    with tab_timing:
        st.caption("Where the time goes. Extraction and quality are LLM calls; "
                   "classification is a single forward pass through RoBERTa.")
        for stage, seconds in timings.items():
            st.markdown(f"**{stage}** — {seconds:.1f}s")
            st.progress(seconds / max(sum(timings.values()), 1e-9))


if __name__ == "__main__":
    main()
