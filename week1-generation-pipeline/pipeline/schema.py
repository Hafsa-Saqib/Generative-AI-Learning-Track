from pydantic import BaseModel, Field


class SourceCitation(BaseModel):
    source_id: str
    clause_or_case: str


class ExceptionCommunication(BaseModel):
    exception_id: str
    customer_facing_draft: str
    internal_summary: str
    grounded_commitments: list[str]
    citations: list[SourceCitation]
    confidence: float = Field(ge=0.0, le=1.0)
    requires_human_review: bool
    review_reason: str | None = None