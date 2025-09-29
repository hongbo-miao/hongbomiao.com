import logging

import httpx
from agent.models.chat_dependencies import ChatAgentDependencies
from agent.models.chat_response import ChatResponse
from agent.utils.chat_agent import chat_agent
from shared.lance_db.models.document_lance_db_context import DocumentLanceDbContext
from shared.openai.utils.build_prompt import build_prompt

logger = logging.getLogger(__name__)


async def get_answer(
    document_context: DocumentLanceDbContext | None,
    httpx_client: httpx.AsyncClient,
    question: str,
) -> ChatResponse:
    try:
        result = await chat_agent.run(
            build_prompt(question),
            deps=ChatAgentDependencies(
                document_context=document_context,
                httpx_client=httpx_client,
            ),
        )
    except Exception as e:
        logger.exception("Error in get_answer")
        return ChatResponse(
            answer=f"I encountered an error while processing your question: {e!s}",
            relevant_chunks=[],
            confidence_scores=[],
            source_pdf_set=set(),
        )
    else:
        return result.output
