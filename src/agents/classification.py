"""Classification agent placeholder.

Day 11 replaces this with inference from the fine-tuned BERT multi-label
classifier. For now it's a pass-through no-op so the LangGraph pipeline has a
second node to route through.
"""

from src.models.requirement import Requirement


def classify_requirements(requirements: list[Requirement]) -> list[Requirement]:
    return requirements
