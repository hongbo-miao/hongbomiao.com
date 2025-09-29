from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    id: str
    object: str = Field(default="model")
    created: int
    owned_by: str = Field(default="hongbomiao")
