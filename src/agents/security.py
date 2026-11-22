"""Security and privacy agent — missing controls and a sensitive-data inventory.

PS agent table: "identifies cybersecurity and privacy requirements". §10's last
quality check: "missing security or compliance controls". Two outputs:

  CONTROLS        The control catalogue (`data/security/controls.yaml`) run
                  through the same rule engine as the compliance pack. A control
                  that applies to this project and that no requirement covers
                  comes back as a gap with a proposed requirement. Controls with
                  no source in the knowledge base are `basis: practice` and say
                  so, rather than borrowing a regulation's authority.

  DATA INVENTORY  Which classes of sensitive data the project touches (Aadhaar,
                  PAN, card data, biometrics, financial profile...), where each
                  is mentioned, and which requirements protect it. Unprotected
                  high-sensitivity data is the finding. The same inventory is
                  what Phase E's masking will need: it is the list of what to
                  mask before text reaches the LLM.

No LLM. A model asked "what security controls are missing?" produces a
plausible list that changes every run; the catalogue produces the same list,
traceable term by term, and grows by editing YAML.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from src.agents.rulepack import evaluate_pack, load_rules
from src.models.analysis import DataItem, SecurityReport
from src.models.requirement import Requirement
from src.utils.evidence import find_terms
from src.utils.similarity import Similarity

REPO = Path(__file__).resolve().parents[2]
CONTROLS_PATH = REPO / "data/security/controls.yaml"


@lru_cache(maxsize=1)
def load_controls():
    return load_rules(CONTROLS_PATH)


@lru_cache(maxsize=1)
def load_inventory_spec() -> dict:
    data = yaml.safe_load(CONTROLS_PATH.read_text(encoding="utf-8")) or {}
    spec = data.get("sensitive_data") or {}
    if not spec.get("classes") or not spec.get("protection_terms"):
        raise ValueError(f"{CONTROLS_PATH.name}: sensitive_data needs classes "
                         f"and protection_terms")
    return spec


def data_inventory(scenario: str, requirements: list[Requirement],
                   spec: dict | None = None) -> list[DataItem]:
    """Sensitive data classes mentioned anywhere, and what protects each."""
    spec = spec or load_inventory_spec()
    protection = spec["protection_terms"]
    blanket = spec.get("blanket_terms", [])
    texts = {r.id: f"{r.statement} {r.acceptance_criteria or ''}" for r in requirements}

    # A requirement that protects "all personal data" protects every class.
    blanket_protectors = [rid for rid, t in texts.items()
                          if find_terms(t, blanket) and find_terms(t, protection)]

    items = []
    for c in spec["classes"]:
        found = ["scenario"] if find_terms(scenario, c["terms"]) else []
        found += [rid for rid, t in texts.items() if find_terms(t, c["terms"])]
        if not found:
            continue
        protected_by = [rid for rid, t in texts.items()
                        if find_terms(t, c["terms"]) and find_terms(t, protection)]
        protected_by += [rid for rid in blanket_protectors if rid not in protected_by]
        items.append(DataItem(kind=c["kind"], label=c["label"],
                              sensitivity=c["sensitivity"], found_in=found,
                              protected_by=protected_by))
    # Unprotected and high-sensitivity first: that is the order a reviewer needs.
    return sorted(items, key=lambda d: (d.protected, d.sensitivity != "high", d.kind))


def analyse_security(scenario: str, requirements: list[Requirement],
                     sim: Similarity | None = None,
                     use_rag: bool = True) -> SecurityReport:
    return SecurityReport(
        controls=evaluate_pack(load_controls(), scenario, requirements,
                               sim=sim, resolve=use_rag),
        data_inventory=data_inventory(scenario, requirements),
    )
