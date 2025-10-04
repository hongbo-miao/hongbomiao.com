from dataclasses import dataclass

import httpx
from shared.lance_db.models.document_lance_db_context import DocumentLanceDbContext


@dataclass
class ChatAgentDependencies:
    httpx_client: httpx.AsyncClient
    document_context: DocumentLanceDbContext | None = None
