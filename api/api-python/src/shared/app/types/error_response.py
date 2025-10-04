from pydantic import BaseModel
from shared.app.types.error_detail import ErrorDetail


class ErrorResponse(BaseModel):
    error: ErrorDetail
