from pydantic import BaseModel, Field
from shared.openai.types.chat_message import ChatMessage


class ChatCompletionRequest(BaseModel):
    model: str = Field()
    messages: list[ChatMessage] = Field()
    temperature: float | None = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)
    stream: bool | None = Field(default=False)
    top_p: float | None = Field(default=1.0, ge=0.0, le=1.0)
    frequency_penalty: float | None = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float | None = Field(default=0.0, ge=-2.0, le=2.0)
    stop: str | list[str] | None = Field(default=None)
    user_id: str | None = Field(default=None)
