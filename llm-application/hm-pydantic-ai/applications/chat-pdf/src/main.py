import asyncio
import logging
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from config import Config
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    PdfPipelineOptions,
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from pydantic import BaseModel, ConfigDict
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class PDFContext(BaseModel):
    chunks: list[str]
    faiss_index: Any = None  # FAISS index can't be serialized
    embeddings: Any = None  # NumPy arrays
    model: Any = None  # SentenceTransformer model
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ChatResponse(BaseModel):
    answer: str
    relevant_chunks: list[str] = []
    confidence_scores: list[float] = []


def create_openai_model() -> OpenAIModel:
    config = Config()
    return OpenAIModel(
        model_name="claude-sonnet-4",
        provider=OpenAIProvider(
            base_url="https://litellm.hongbomiao.com/v1",
            api_key=config.OPENAI_API_KEY,
        ),
    )


def process_pdf(pdf_path: Path) -> list[str]:
    """Process PDF and extract text chunks."""
    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=True,
        table_structure_options=TableStructureOptions(do_cell_matching=True),
        ocr_options=EasyOcrOptions(),
    )
    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        },
    )
    logger.info("Converting PDF to Markdown")
    res = converter.convert(pdf_path)
    markdown_content = res.document.export_to_markdown()
    # Split the markdown text into chunks
    chunks = []
    current_chunk: list[str] = []
    current_length = 0
    max_chunk_length = 1000
    for line in markdown_content.split("\n"):
        line_length = len(line)
        if current_length + line_length > max_chunk_length and current_chunk:
            chunks.append("\n".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(line)
        current_length += line_length
    if current_chunk:
        chunks.append("\n".join(current_chunk))
    return chunks


def set_up_vector_store(texts: list[str]) -> PDFContext:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    # Create embeddings for all chunks
    embeddings = model.encode(texts)
    embeddings = np.array(embeddings, dtype=np.float32)
    faiss.normalize_L2(embeddings)
    # Create FAISS index
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return PDFContext(
        chunks=texts,
        faiss_index=index,
        embeddings=embeddings,
        model=model,
    )


chat_agent = Agent(
    model=create_openai_model(),
    output_type=ChatResponse,
    system_prompt="""
    You are a helpful AI assistant that answers questions based on PDF document content.
    You have access to a tool called 'retrieve_relevant_context' that can search through
    the PDF content to find relevant information.
    When a user asks a question:
    1. First use the retrieve_relevant_context tool to find relevant information
    2. Then provide a comprehensive answer based on that context
    3. If the context doesn't contain enough information, say so clearly
    4. Always be precise, helpful, and cite specific parts of the context when relevant
    Format your response as a ChatResponse with a detailed answer.
    """,
)


@chat_agent.tool
def retrieve_relevant_context(
    ctx: RunContext[PDFContext],
    query: str,
) -> dict[str, Any]:
    """Retrieve relevant context from the PDF using vector similarity search."""
    try:
        pdf_context = ctx.deps
        # Encode the query
        query_embedding = pdf_context.model.encode([query])
        query_embedding = np.array(query_embedding, dtype=np.float32)
        faiss.normalize_L2(query_embedding)
        # Search for similar chunks
        k = min(5, len(pdf_context.chunks))  # Number of chunks to retrieve
        scores, indices = pdf_context.faiss_index.search(query_embedding, k)
        relevant_chunks = [pdf_context.chunks[idx] for idx in indices[0]]
        confidence_scores = scores[0].tolist()
        # Combine relevant chunks into context
        context = "\n\n".join(relevant_chunks)
        logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks")
    except Exception:
        logger.exception("Error in retrieve_relevant_context")
        return {
            "context": "Error retrieving context",
            "relevant_chunks": [],
            "confidence_scores": [],
        }
    else:
        return {
            "context": context,
            "relevant_chunks": relevant_chunks,
            "confidence_scores": confidence_scores,
        }


async def ask_question(pdf_context: PDFContext, question: str) -> ChatResponse:
    try:
        # Let the agent handle tool calling automatically
        result = await chat_agent.run(
            f"Please answer this question about the PDF: {question}",
            deps=pdf_context,
        )
    except Exception as e:
        logger.exception("Error in ask_question")
        return ChatResponse(
            answer=f"I encountered an error while processing your question: {e!s}",
            relevant_chunks=[],
            confidence_scores=[],
        )
    else:
        return result.output


async def chat_with_pdf(pdf_path: Path, question: str) -> str:
    try:
        logger.info(f"Processing PDF: {pdf_path}")
        chunks = process_pdf(pdf_path)
        pdf_context = set_up_vector_store(chunks)
        logger.info(f"PDF processed successfully. Created {len(chunks)} chunks.")
        response = await ask_question(pdf_context, question)
    except Exception as e:
        logger.exception("Error in chat_with_pdf")
        return f"Error processing PDF: {e!s}"
    else:
        return response.answer


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    pdf_path = Path("data/file.pdf")
    question = "Could you please summarize this PDF?"
    answer = await chat_with_pdf(pdf_path, question)
    logger.info(f"Answer: {answer}")


if __name__ == "__main__":
    asyncio.run(main())
