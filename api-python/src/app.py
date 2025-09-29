import logging

import sentry_sdk
from config import config
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sentry_sdk.integrations.fastapi import FastApiIntegration
from shared.routers import health, motor
from shared.routers.http_exception_handler import http_exception_handler
from utils.logger import logger

logger.setLevel(logging.INFO)
logger.info(f"{config = }")

sentry_sdk.init(
    dsn=config.SENTRY_DSN,
    integrations=[FastApiIntegration()],
    traces_sample_rate=1.0,
    environment=config.ENV,
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(health.router)
app.include_router(motor.router)
app.add_exception_handler(HTTPException, http_exception_handler)
