from fastapi import Request
from mem0 import Memory


def get_memory_client(request: Request) -> Memory:
    return request.app.state.memory_client
