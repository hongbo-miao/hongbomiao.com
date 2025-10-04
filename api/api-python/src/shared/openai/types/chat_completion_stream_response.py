from pydantic import BaseModel, Field
from shared.openai.types.chat_completion_stream_choice import ChatCompletionStreamChoice


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = Field(default="chat.completion.chunk")
    created: int
    model: str
    choices: list[ChatCompletionStreamChoice]
