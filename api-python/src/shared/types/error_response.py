from pydantic import BaseModel
from shared.types.error_detail import ErrorDetail


class ErrorResponse(BaseModel):
    error: ErrorDetail
