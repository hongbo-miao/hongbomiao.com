from config import config
from mem0 import Memory
from shared.memory.utils.search_memories import search_memories


def build_prompt(memory_client: Memory, question: str, user_id: str | None) -> str:
    memories = search_memories(
        memory_client=memory_client,
        query=question,
        user_id=user_id,
        limit=config.MEMORY_LIMIT,
    )
    if memories:
        memories_str = "\n".join(f"- {memory}" for memory in memories)
        memories_block = f"User memories:\n{memories_str}\n\n"
    else:
        memories_block = ""
    return f"{memories_block}{question}"
