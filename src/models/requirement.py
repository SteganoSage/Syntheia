"""Pydantic schemas for extracted software requirements.

Fields follow the structure defined in the project problem statement (Section 8).
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field

Priority = Literal["High", "Medium", "Low"]
RiskLevel = Literal["High", "Medium", "Low"]
ApprovalStatus = Literal["Pending", "Approved", "Rejected"]


class Requirement(BaseModel):
    id: str = Field(description="Requirement identifier, e.g. REQ-001")
    statement: str = Field(
        description="A single, atomic 'The system shall ...' requirement statement"
    )
    category: list[str] = Field(
        default_factory=list,
        description=(
            "Applicable categories from: Business, Stakeholder, Functional, "
            "Security, Privacy, Regulatory, Performance, Availability, "
            "Usability, Data-management, Integration, Audit/Reporting, Operational"
        ),
    )
    source_stakeholder: str = Field(
        default="",
        description="Who stated or needs this requirement, e.g. 'Business Analyst', 'Compliance Officer', 'End User'",
    )
    business_justification: str = Field(
        default="",
        description="Why this requirement matters — the business reasoning behind it",
    )
    priority: Priority = Field(default="Medium")
    dependencies: list[str] = Field(
        default_factory=list,
        description="IDs of other requirements this depends on, e.g. ['REQ-002', 'REQ-005']",
    )
    assumptions: str = Field(
        default="",
        description="What is assumed when stating this requirement",
    )
    acceptance_criteria: str = Field(
        default="",
        description="How to verify this requirement is met — a testable condition",
    )
    applicable_regulations: list[str] = Field(
        default_factory=list,
        description="Relevant regulations, standards, or policies, e.g. ['RBI Digital Lending Guidelines', 'PCI-DSS']",
    )
    risk_level: RiskLevel = Field(
        default="Medium",
        description="Risk level if this requirement is not met",
    )
    confidence_score: float = Field(
        default=0.8,
        description="Model's confidence in this extraction (0.0 to 1.0)",
    )
    approval_status: ApprovalStatus = Field(
        default="Pending",
        description="Whether this requirement has been approved by stakeholders",
    )


class ExtractionResult(BaseModel):
    requirements: list[Requirement]
