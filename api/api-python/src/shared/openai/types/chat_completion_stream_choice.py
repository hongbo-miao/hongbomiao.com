from typing import Any

from pydantic import BaseModel


class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: dict[str, Any]
    finish_reason: str | None = None
