from typing import Literal, Optional

from pydantic import BaseModel, Field

Issue_Type = Literal["Ambiguity","Incompleteness","Inconsistency","Duplication","Infeasibility","Lack of Testability","Missing Source","Undefined Terminology","Unverifiable Requirement","Conflicting Expectations","Missing security/compliance"]
Severity = Literal["Critical","High","Medium","Low"]

class QualityIssue(BaseModel):
    issue_type : Issue_Type = Field(description = "Type of Quality Issue Detected")
    description : str = Field(description = "Description of the quality issue")
    severity : Severity = Field(description = "Severity level of the issue")
    suggestion : str = Field(description = "Suggested fix or improvement")

class QualityResult(BaseModel):
    requirement_id : str = Field(description = "Requirement ID")
    issues : list[QualityIssue] = Field(description = "List of quality issues found")
    passed : bool = Field(description = "Whether the requirement passed all quality checks", default=False)

class QualityReport(BaseModel):
    result_list : list[QualityResult] = Field(default_factory=list, description = "List of quality results")
    