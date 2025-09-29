import asyncio
from collections.abc import AsyncGenerator

import httpx
from shared.lance_db.models.document_lance_db_context import DocumentLanceDbContext
from shared.openai.types.chat_completion_request import ChatCompletionRequest
from shared.openai.types.chat_completion_stream_choice import ChatCompletionStreamChoice
from shared.openai.types.chat_completion_stream_response import (
    ChatCompletionStreamResponse,
)
from shared.openai.utils.generate_completion_meta import generate_completion_meta
from shared.openai.utils.stream_answer import stream_answer


async def create_streaming_completion(
    request: ChatCompletionRequest,
    question: str,
    document_context: DocumentLanceDbContext | None,
    httpx_client: httpx.AsyncClient,
) -> AsyncGenerator[str, None]:
    completion_id, created, _ = generate_completion_meta(request)

    async for delta_text in stream_answer(
        document_context=document_context,
        httpx_client=httpx_client,
        question=question,
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
    yield "data: [DONE]\n\n"
