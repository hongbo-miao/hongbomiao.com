from pydantic import BaseModel
from shared.openai.types.error_detail import ErrorDetail


class ErrorResponse(BaseModel):
    error: ErrorDetail
