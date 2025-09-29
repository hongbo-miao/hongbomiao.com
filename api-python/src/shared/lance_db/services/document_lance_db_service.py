import logging
from pathlib import Path

import lancedb
from shared.lance_db.models.document_lance_db_context import DocumentLanceDbContext
from shared.llm_model.utils.load_embedding_model import load_embedding_model

logger = logging.getLogger(__name__)


class DocumentLanceDbService:
    @staticmethod
    def load_document_lance_db(
        document_lance_db_dir_path: Path,
    ) -> DocumentLanceDbContext | None:
        lance_db_folder = document_lance_db_dir_path / "pdf_chunks.lance"
        if not lance_db_folder.exists():
            message = "No database found."
            logger.warning(message)
            return None

        try:
            document_lance_db = lancedb.connect(document_lance_db_dir_path)
            table_name = "pdf_chunks"
            table = document_lance_db.open_table(table_name)

            # Get all chunks to provide context about available data
            all_chunks = table.search().limit(10000).to_list()
            chunks = [chunk["text"] for chunk in all_chunks]
            processed_pdf_set = {chunk["pdf_path"] for chunk in all_chunks}

            logger.info(
                f"Loaded existing database with {len(chunks)} chunks from {len(processed_pdf_set)} PDFs: {processed_pdf_set}",
            )

            model = load_embedding_model()
            return DocumentLanceDbContext(
                chunks=chunks,
                lance_table=table,
                model=model,
                processed_pdf_set=processed_pdf_set,
            )
        except Exception:
            logger.exception("Error loading existing database.")
            message = "No existing database found."
            raise ValueError(message) from None
