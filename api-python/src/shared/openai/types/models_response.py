from pydantic import BaseModel, Field
from shared.openai.types.model_info import ModelInfo


class ModelsResponse(BaseModel):
    object: str = Field(default="list")
    data: list[ModelInfo]
