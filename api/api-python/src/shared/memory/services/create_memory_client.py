from typing import Any

from config import config
from mem0 import Memory


def create_memory_client() -> Memory:
    memory_config: dict[str, Any] = {
        "llm": {
            "provider": "openai",
            "config": {
                "model": config.MEMORY_MODEL,
                "temperature": config.MEMORY_MODEL_TEMPERATURE,
                "max_tokens": config.MEMORY_MODEL_MAX_TOKENS,
                "api_key": config.OPENAI_API_KEY,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "api_key": config.OPENAI_API_KEY,
            },
        },
    }
    return Memory.from_config(memory_config)
