from pydantic import BaseModel


class ErrorDetail(BaseModel):
    message: str
    type: str
    param: str | None = None
    code: str | None = None
