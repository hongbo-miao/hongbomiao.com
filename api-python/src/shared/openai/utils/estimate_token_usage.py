from shared.openai.types.chat_message import ChatMessage


def estimate_token_usage(
    messages: list[ChatMessage],
    answer: str,
) -> tuple[int, int, int]:
    prompt_tokens = sum(len(message.content.split()) for message in messages)
    completion_tokens = len(answer.split())
    total_tokens = prompt_tokens + completion_tokens
    return prompt_tokens, completion_tokens, total_tokens
