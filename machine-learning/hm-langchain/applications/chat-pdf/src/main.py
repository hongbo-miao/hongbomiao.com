import logging

from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import GPT4AllEmbeddings
from langchain.llms import GPT4All
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma


def main(model_path: str, pdf_path: str, question: str) -> None:
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=30)
    docs = text_splitter.split_documents(documents)
    embeddings = GPT4AllEmbeddings()
    vectorstore = Chroma.from_documents(docs, embeddings)
    llm = GPT4All(model=model_path, max_tokens=2048)
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
    )
    answer = qa.run(question)
    logging.info(answer)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # https://gpt4all.io/index.html
    external_model_path = "data/ggml-model-gpt4all-falcon-q4_0.bin"
    external_pdf_path = "data/my.pdf"
    external_question = "Could you please summarize this PDF? Thank you!"

    main(external_model_path, external_pdf_path, external_question)
