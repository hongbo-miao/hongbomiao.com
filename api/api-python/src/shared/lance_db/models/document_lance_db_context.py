import lancedb
from pydantic import BaseModel, ConfigDict
from sentence_transformers import SentenceTransformer


class DocumentLanceDbContext(BaseModel):
    chunks: list[str]
    lance_table: lancedb.db.Table | None = None
    model: SentenceTransformer | None = None
    processed_pdf_set: set[str] = set()
    model_config = ConfigDict(arbitrary_types_allowed=True)
