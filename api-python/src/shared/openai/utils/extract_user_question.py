from fastapi import HTTPException
from shared.openai.types.chat_completion_request import ChatCompletionRequest
from shared.openai.types.role import Role


def extract_user_question(request: ChatCompletionRequest) -> str:
    user_messages = [
        message for message in request.messages if message.role == Role.USER
    ]
    if not user_messages:
        raise HTTPException(
            status_code=400,
            detail="No user message found in request",
        )
    return user_messages[-1].content
