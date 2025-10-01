from mem0 import Memory


def add_conversation_to_memory(
    memory_client: Memory,
    user_message: str,
    assistant_message: str,
    user_id: str,
) -> None:
    messages = [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_message},
    ]
    memory_client.add(messages, user_id=user_id)
