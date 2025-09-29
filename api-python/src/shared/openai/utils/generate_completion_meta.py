import time
import uuid

from shared.openai.types.chat_completion_request import ChatCompletionRequest


def generate_completion_meta(request: ChatCompletionRequest) -> tuple[str, int, str]:
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
    created = int(time.time())
    return completion_id, created, request.model
