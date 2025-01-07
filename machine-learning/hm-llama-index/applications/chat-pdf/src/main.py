import logging
from pathlib import Path

from config import Config
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.litellm import LiteLLM

logger = logging.getLogger(__name__)


def chat_with_pdf(pdf_path: Path, question: str) -> str:
    documents = SimpleDirectoryReader(input_files=[pdf_path]).load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    res = query_engine.query(question)
    return res.response


def main() -> None:
    config = Config()

    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )
    Settings.llm = LiteLLM(
        api_base="https://litellm.internal.hongbomiao.com",
        api_key=config.OPENAI_API_KEY,
        model="openai/claude-3-5-sonnet",
        temperature=0.7,
    )

    pdf_path = Path("data/file.pdf")
    question = "Could you please summarize this PDF?"
    answer = chat_with_pdf(pdf_path, question)
    logger.info(answer)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
