"""Pydantic schemas for extracted software requirements."""

from typing import Literal

from pydantic import BaseModel, Field

Priority = Literal["High", "Medium", "Low"]


class Requirement(BaseModel):
    id: str = Field(description="Requirement identifier, e.g. REQ-001")
    statement: str = Field(
        description="A single, atomic 'The system shall ...' requirement statement"
    )
    priority: Priority
    category_hint: str = Field(
        default="",
        description=(
            "Comma-separated best-guess categories for this requirement, "
            "e.g. 'Security, Regulatory'"
        ),
    )


class ExtractionResult(BaseModel):
    requirements: list[Requirement]
