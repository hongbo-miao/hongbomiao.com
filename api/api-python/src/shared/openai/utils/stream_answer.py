import asyncio
import logging
from collections.abc import AsyncGenerator

import httpx
from agent.models.chat_dependencies import ChatAgentDependencies
from agent.utils.chat_agent import chat_agent
from mem0 import Memory
from shared.lance_db.models.document_lance_db_context import DocumentLanceDbContext
from shared.openai.utils.build_prompt import build_prompt

logger = logging.getLogger(__name__)


async def stream_answer(
    memory_client: Memory,
    document_context: DocumentLanceDbContext | None,
    httpx_client: httpx.AsyncClient,
    question: str,
    user_id: str | None,
) -> AsyncGenerator[str, None]:
    try:
        async with chat_agent.run_stream(
            build_prompt(memory_client, question, user_id),
            deps=ChatAgentDependencies(
                document_context=document_context,
                httpx_client=httpx_client,
            ),
        ) as response:
            prev_text = ""
            async for item in response.stream(debounce_by=0.01):
                current_text = item.answer
                if not current_text:
                    continue

                # Emit only the delta from the last seen content
                if current_text.startswith(prev_text):
                    delta_text = current_text[len(prev_text) :]
                else:
                    # Fallback if the stream resets or diverges
                    delta_text = current_text
                prev_text = current_text

                if not delta_text:
                    continue

                yield delta_text
                # Flush
                await asyncio.sleep(0)
    except Exception:
        logger.exception("Error in stream_answer")
        yield "I encountered an error while streaming the answer. Please try again."
