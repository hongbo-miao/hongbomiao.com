from pydantic import BaseModel
from shared.openai.types.role import Role


class ChatMessage(BaseModel):
    role: Role
    content: str
