from config import Config
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.litellm import LiteLLM


def chat_with_pdf(pdf_path, question):
    config = Config()

    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    Settings.llm = LiteLLM(
        api_base="https://litellm.internal.hongbomiao.com",
        api_key=config.OPENAI_API_KEY,
        model="openai/claude-3-5-sonnet",
    )

    documents = SimpleDirectoryReader(input_files=[pdf_path]).load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()

    response = query_engine.query(question)
    return response.response


if __name__ == "__main__":
    pdf_path = "data/paper.pdf"
    question = "Could you please summarize this PDF? Thank you!"
    summary = chat_with_pdf(pdf_path, question)
    print(summary)
