import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from shared.lance_db.models.document_lance_db_context import DocumentLanceDbContext
from shared.openai.types.chat_completion_request import ChatCompletionRequest
from shared.openai.types.chat_completion_response import ChatCompletionResponse
from shared.openai.utils.create_non_streaming_completion import (
    create_non_streaming_completion,
)
from shared.openai.utils.create_streaming_completion import create_streaming_completion
from shared.openai.utils.extract_user_question import extract_user_question

logger = logging.getLogger(__name__)

router = APIRouter(tags=["completions"])


@router.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    fastapi_request: Request,
) -> ChatCompletionResponse:
    try:
        document_context: DocumentLanceDbContext | None = (
            fastapi_request.app.state.document_context
        )
        question = extract_user_question(request)
        if request.stream:
            return StreamingResponse(
                create_streaming_completion(
                    request,
                    question,
                    document_context,
                    fastapi_request.app.state.httpx_client,
                ),
                media_type="text/event-stream",
                headers={
                    # Prevent intermediaries from buffering server-sent event (SSE)
                    "Cache-Control": "no-cache, no-store, no-transform",
                },
            )
        return await create_non_streaming_completion(
            request,
            question,
            document_context,
            httpx_client=fastapi_request.app.state.httpx_client,
        )

    except Exception:
        logger.exception("Error in create_chat_completion")
        raise HTTPException(
            status_code=500,
            detail="Internal server error.",
        ) from None
