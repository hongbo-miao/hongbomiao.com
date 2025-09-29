from pydantic import BaseModel, Field


class ChatResponse(BaseModel):
    # Provide defaults so partial streaming validations succeed
    answer: str = ""
    relevant_chunks: list[str] = Field(default_factory=list)
    confidence_scores: list[float] = Field(default_factory=list)
    source_pdf_set: set[str] = Field(default_factory=set)
