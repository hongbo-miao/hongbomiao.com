import asyncio
from collections.abc import AsyncGenerator

import httpx
from mem0 import Memory
from shared.lance_db.models.document_lance_db_context import DocumentLanceDbContext
from shared.memory.utils.add_conversation_to_memory import add_conversation_to_memory
from shared.openai.types.chat_completion_request import ChatCompletionRequest
from shared.openai.types.chat_completion_stream_choice import ChatCompletionStreamChoice
from shared.openai.types.chat_completion_stream_response import (
    ChatCompletionStreamResponse,
)
from shared.openai.utils.generate_completion_meta import generate_completion_meta
from shared.openai.utils.stream_answer import stream_answer


async def create_streaming_completion(
    memory_client: Memory,
    request: ChatCompletionRequest,
    question: str,
    document_context: DocumentLanceDbContext | None,
    httpx_client: httpx.AsyncClient,
) -> AsyncGenerator[str, None]:
    completion_id, created, _ = generate_completion_meta(request)
    full_answer = ""
    async for delta_text in stream_answer(
        memory_client=memory_client,
        document_context=document_context,
        httpx_client=httpx_client,
        question=question,
        user_id=request.user_id,
    ):
        if not delta_text:
            continue

        stream_response = ChatCompletionStreamResponse(
            id=completion_id,
            created=created,
            model=request.model,
            choices=[
                ChatCompletionStreamChoice(
                    index=0,
                    delta={"content": delta_text},
                ),
            ],
        )
        full_answer += delta_text
        yield f"data: {stream_response.model_dump_json()}\n\n"
        # Flush
        await asyncio.sleep(0)

    # Send final chunk with finish_reason
    final_response = ChatCompletionStreamResponse(
        id=completion_id,
        created=created,
        model=request.model,
        choices=[
            ChatCompletionStreamChoice(
                index=0,
                delta={},
                finish_reason="stop",
            ),
        ],
    )

    yield f"data: {final_response.model_dump_json()}\n\n"
    # Flush
    await asyncio.sleep(0)
    add_conversation_to_memory(
        memory_client=memory_client,
        user_message=question,
        assistant_message=full_answer,
        user_id=request.user_id,
    )
    yield "data: [DONE]\n\n"
