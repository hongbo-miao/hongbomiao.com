import logging

from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.llms import GPT4All
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def main(model_path: str, pdf_path: str, question: str) -> None:
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=30)
    docs = text_splitter.split_documents(documents)
    embeddings = GPT4AllEmbeddings()
    vectorstore = Chroma.from_documents(docs, embeddings)
    llm = GPT4All(model=model_path, max_tokens=2048)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
    )
    answer = qa.run(question)
    logger.info(answer)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # https://gpt4all.io/index.html
    external_model_path = "data/ggml-model-gpt4all-falcon-q4_0.bin"
    external_pdf_path = "data/my.pdf"
    external_question = "Could you please summarize this PDF?"

    main(external_model_path, external_pdf_path, external_question)
