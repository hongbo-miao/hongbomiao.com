from pydantic import BaseModel
from shared.openai.types.chat_message import ChatMessage


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str | None = "stop"
