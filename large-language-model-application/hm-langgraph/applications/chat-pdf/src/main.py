import logging
from pathlib import Path

import faiss
from config import config
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    PdfPipelineOptions,
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from langgraph.graph import Graph, MessagesState  # type: ignore[attr-defined]
from openai import OpenAI
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def create_openai_client() -> OpenAI:
    return OpenAI(
        base_url="https://litellm.hongbomiao.com/v1",
        api_key=config.OPENAI_API_KEY,
    )


def process_pdf(pdf_path: Path) -> list[str]:
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


def setup_vector_store(
    texts: list[str],
) -> tuple[faiss.IndexFlatIP, list[str], SentenceTransformer]:  # type: ignore[type-arg]
    # Initialize the sentence transformer model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Create embeddings for all chunks
    embeddings = model.encode(texts)
    faiss.normalize_L2(embeddings)

    # Create FAISS index
    index = faiss.IndexFlatIP(embeddings.shape[1])  # type: ignore[attr-defined]
    index.add(embeddings)  # type: ignore[arg-type]

    return index, embeddings, model  # type: ignore[return-value]


def retrieve_context(
    state: MessagesState,
    index: faiss.IndexFlatIP,  # type: ignore
    chunks: list[str],
    model: SentenceTransformer,
) -> MessagesState:
    try:
        logger.info("Starting context retrieval")
        query_embedding = model.encode([state["question"]])  # type: ignore[index]
        faiss.normalize_L2(query_embedding)

        k = 3  # Number of chunks to retrieve
        _scores, indices = index.search(query_embedding, k)  # type: ignore[arg-type]

        relevant_chunks = [chunks[idx] for idx in indices[0]]
        state["context"] = "\n".join(relevant_chunks)  # type: ignore[index]
        logger.info("Context retrieval completed")
    except Exception:
        logger.exception("Error in retrieve_context.")
        raise
    else:
        return state


def generate_answer(state: MessagesState) -> MessagesState:
    try:
        client = create_openai_client()
        response = client.chat.completions.create(
            model="claude-opus-4-5",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on the provided context.",
                },
                {
                    "role": "user",
                    "content": f"""
                        Answer the following question based on the provided context:
                        Context: {state["context"]}
                        Question: {state["question"]}
                    """,  # type: ignore[index]
                },
            ],
        )
        state["answer"] = response.choices[0].message.content  # type: ignore[index]
    except Exception:
        logger.exception("Error in generate_answer.")
        raise
    else:
        return state


def create_graph(
    index: faiss.IndexFlatIP,  # type: ignore[type-arg]
    chunks: list[str],
    model: SentenceTransformer,
) -> Graph:
    graph = (
        Graph()
        .add_node(
            "retrieve",
            lambda state: retrieve_context(state, index, chunks, model),
        )
        .add_node("answer", generate_answer)
        .add_edge("retrieve", "answer")
        .set_entry_point("retrieve")
        .set_finish_point("answer")
    )
    return graph.compile()


def chat_with_pdf(pdf_path: Path, question: str) -> str:
    try:
        # Process PDF and setup vector store
        chunks = process_pdf(pdf_path)
        index, _embeddings, model = setup_vector_store(chunks)

        # Create initial state
        initial_state = MessagesState(  # ty: ignore[missing-typed-dict-key]
            {"question": question, "context": "", "answer": ""},  # ty:ignore[invalid-key]
        )

        # Create and run graph
        graph = create_graph(index, chunks, model)
        for event in graph.stream(initial_state):
            if "answer" in event:
                return event["answer"]
    except Exception:
        logger.exception("Error in chat_with_pdf")
        raise
    else:
        return "No answer was found for the given question"


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    pdf_path = Path("data/file.pdf")
    question = "Could you please summarize this PDF?"
    answer = chat_with_pdf(pdf_path, question)
    logger.info(answer)
