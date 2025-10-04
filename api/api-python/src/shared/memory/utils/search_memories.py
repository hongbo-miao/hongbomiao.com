from typing import Any

from mem0 import Memory


def search_memories(
    memory_client: Memory,
    query: str,
    user_id: str,
    limit: int,
) -> list[str]:
    results: dict[str, Any] = memory_client.search(
        query=query,
        user_id=user_id,
        limit=limit,
    )
    entries = results.get("results", [])
    return [memory for entry in entries if (memory := entry.get("memory"))]
