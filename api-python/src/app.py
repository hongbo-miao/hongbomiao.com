from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import sentry_sdk
from config import config
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from routers import completions, health, models, motor
from routers.handle_http_exception import handle_http_exception
from sentry_sdk.integrations.fastapi import FastApiIntegration
from shared.lance_db.services.document_lance_db_service import DocumentLanceDbService

sentry_sdk.init(
    dsn=config.SENTRY_DSN,
    integrations=[FastApiIntegration()],
    traces_sample_rate=1.0,
    environment=config.ENV,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    app.state.document_context = DocumentLanceDbService.load_document_lance_db(
        Path(config.DOCUMENT_LANCE_DB_DIR),
    )
    app.state.httpx_client = httpx.AsyncClient(timeout=30.0)
    yield
    if app.state.document_context:
        del app.state.document_context
    await app.state.httpx_client.aclose()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_headers=["*"],
    allow_methods=["GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"],
    allow_origins=["*"],
)
app.include_router(completions.router)
app.include_router(health.router)
app.include_router(models.router)
app.include_router(motor.router)
app.add_exception_handler(HTTPException, handle_http_exception)
