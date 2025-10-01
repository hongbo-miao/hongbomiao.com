import httpx
from mem0 import Memory
from shared.lance_db.models.document_lance_db_context import DocumentLanceDbContext
from shared.memory.utils.add_conversation_to_memory import add_conversation_to_memory
from shared.openai.types.chat_completion_choice import ChatCompletionChoice
from shared.openai.types.chat_completion_request import ChatCompletionRequest
from shared.openai.types.chat_completion_response import ChatCompletionResponse
from shared.openai.types.chat_message import ChatMessage
from shared.openai.types.role import Role
from shared.openai.types.usage import Usage
from shared.openai.utils.estimate_token_usage import estimate_token_usage
from shared.openai.utils.generate_completion_meta import generate_completion_meta
from shared.openai.utils.get_answer import get_answer


async def create_non_streaming_completion(
    memory_client: Memory,
    request: ChatCompletionRequest,
    question: str,
    document_context: DocumentLanceDbContext | None,
    httpx_client: httpx.AsyncClient,
) -> ChatCompletionResponse:
    response = await get_answer(
        memory_client,
        document_context,
        httpx_client,
        question,
        request.user_id,
    )
    assistant_message = ChatMessage(
        role=Role.ASSISTANT,
        content=response.answer,
    )
    completion_id, created, model = generate_completion_meta(request)
    prompt_tokens, completion_tokens, total_tokens = estimate_token_usage(
        request.messages,
        response.answer,
    )
    add_conversation_to_memory(
        memory_client=memory_client,
        user_message=question,
        assistant_message=response.answer,
        user_id=request.user_id,
    )
    return ChatCompletionResponse(
        id=completion_id,
        created=created,
        model=model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=assistant_message,
                finish_reason="stop",
            ),
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        ),
    )
