from pydantic import BaseModel, Field
from shared.openai.types.chat_completion_choice import ChatCompletionChoice
from shared.openai.types.usage import Usage


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = Field(default="chat.completion")
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage
