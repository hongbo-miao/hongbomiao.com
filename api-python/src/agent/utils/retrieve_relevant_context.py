import logging
from typing import Any

import numpy as np
from agent.models.chat_dependencies import ChatAgentDependencies
from pydantic_ai import RunContext
from pydantic_ai.exceptions import AgentRunError

logger = logging.getLogger(__name__)


def retrieve_relevant_context(
    ctx: RunContext[ChatAgentDependencies],
    query: str,
) -> dict[str, Any]:
    try:
        document_context = ctx.deps.document_context
        if document_context is None:
            return {
                "context": "No PDFs available in the database",
                "relevant_chunks": [],
                "confidence_scores": [],
                "source_pdf_set": set(),
            }

        # Check the model
        if document_context.model is None:
            return {
                "context": "Embedding model is not available",
                "relevant_chunks": [],
                "confidence_scores": [],
                "source_pdf_set": set(),
            }

        # Encode the query
        query_embedding = document_context.model.encode([query])
        query_embedding = np.array(query_embedding, dtype=np.float32)

        # Search for similar chunks using LanceDB
        k = min(10, len(document_context.chunks))
        results = (
            document_context.lance_table.search(query_embedding[0]).limit(k).to_list()
        )

        relevant_chunks = [result["text"] for result in results]
        source_pdf_set = {result["pdf_path"] for result in results}

        # Convert distances to similarity scores
        confidence_scores = [1.0 / (1.0 + result["_distance"]) for result in results]

        # Combine relevant chunks into context
        context_parts = []
        for chunk, pdf_name in zip(
            relevant_chunks,
            [result["pdf_path"] for result in results],
            strict=False,
        ):
            context_parts.append(f"From {pdf_name}:\n{chunk}")

        context = "\n\n" + "=" * 50 + "\n\n".join(context_parts)

        logger.info(
            f"Retrieved {len(relevant_chunks)} relevant chunks from {len(source_pdf_set)} PDFs: {source_pdf_set}",
        )

    except Exception as error:
        message = "Error in retrieve_relevant_context"
        raise AgentRunError(message) from error
    else:
        return {
            "context": context,
            "relevant_chunks": relevant_chunks,
            "confidence_scores": confidence_scores,
            "source_pdf_set": source_pdf_set,
        }
